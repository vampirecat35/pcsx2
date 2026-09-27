// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/Common/GSDLSSNR.h"

#include "common/Console.h"

#ifdef ENABLE_DLSSNR

#include "nr_frame.h"

#include "common/Threading.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace
{
	// Only touched by the worker thread, which owns the model.
	struct DLSSNRState
	{
		nr_frame* frame = nullptr;
		bool error_reported = false;
		int frame_index = 0;

		u32 width = 0;
		u32 height = 0;
		bool have_history = false;

		std::vector<float> colour;
		std::vector<float> output;
		std::vector<float> history; // previous output
		std::vector<float> previous; // previous input
	};

	// Shared between the GS thread and the worker, guarded by mutex.
	struct DLSSNRWorker
	{
		std::thread thread;
		std::mutex mutex;
		std::condition_variable cv;
		bool quit = false;

		bool job_pending = false;
		std::vector<u8> job;
		u32 job_width = 0;
		u32 job_height = 0;
		float job_intensity = 1.0f;

		bool result_ready = false;
		std::vector<u8> result;
		u32 result_width = 0;
		u32 result_height = 0;

		std::atomic_bool busy{false};
		std::atomic_bool failed{false};
		std::atomic_bool reset_history{false};
	};

	static DLSSNRState s_state;
	static DLSSNRWorker s_worker;
} // namespace

bool GSDLSSNR::IsAvailable()
{
	return true;
}

static bool OpenFrame()
{
	if (s_state.frame)
		return true;

	// NULL: the weights compiled into libdlssnr.
	s_state.frame = nr_frame_open(nullptr);
	if (!s_state.frame)
	{
		Console.Error("DLSS-NR: nr_frame_open() failed: %s", nr_frame_error());
		return false;
	}

	Console.WriteLn("DLSS-NR: running on %s (%s runtime, %s)", nr_frame_device(s_state.frame), nr_frame_runtime(),
		nr_frame_gemm_path(s_state.frame));
	return true;
}

static void CloseFrame()
{
	if (s_state.frame)
	{
		nr_frame_close(s_state.frame);
		nr_frame_shutdown();
		s_state.frame = nullptr;
	}
	s_state.have_history = false;
	s_state.error_reported = false;
}

// Filters an RGBA8 image (width * 4 stride) in place, on the worker thread.
static bool ProcessFrame(u8* rgba, u32 width, u32 height, float intensity)
{
	const size_t pixels = static_cast<size_t>(width) * height;
	const u32 stride = width * 4;
	if (width != s_state.width || height != s_state.height)
	{
		s_state.width = width;
		s_state.height = height;
		s_state.have_history = false;
		s_state.colour.resize(pixels * 3);
		s_state.output.resize(pixels * 3);
		s_state.history.resize(pixels * 3);
		s_state.previous.resize(pixels * 3);
	}

	// byte / 255, like image_io.py and nr_frame_main.c
	for (u32 y = 0; y < height; y++)
	{
		const u8* row = rgba + static_cast<size_t>(y) * stride;
		float* out = s_state.colour.data() + static_cast<size_t>(y) * width * 3;
		for (u32 x = 0; x < width; x++)
		{
			out[x * 3 + 0] = static_cast<float>(row[x * 4 + 0]) * (1.0f / 255.0f);
			out[x * 3 + 1] = static_cast<float>(row[x * 4 + 1]) * (1.0f / 255.0f);
			out[x * 3 + 2] = static_cast<float>(row[x * 4 + 2]) * (1.0f / 255.0f);
		}
	}

	nr_frame_params params;
	nr_frame_defaults(&params);
	params.intensity = intensity;
	params.frame_index = s_state.frame_index++;
	// A PS2 frame is small, run the graph at its own size rather than padding it out to 320.
	params.min_extent = 128;

	const float* history = s_state.have_history ? s_state.history.data() : nullptr;
	const float* previous = s_state.have_history ? s_state.previous.data() : nullptr;
	if (nr_frame_update(s_state.frame, s_state.colour.data(), static_cast<int>(height), static_cast<int>(width), history,
			previous, &params, s_state.output.data(), nullptr) != 0)
	{
		if (!s_state.error_reported)
		{
			Console.Error("DLSS-NR: nr_frame_update() failed: %s", nr_frame_error());
			s_state.error_reported = true;
		}
		return false;
	}

	s_state.history.swap(s_state.output);
	s_state.previous.swap(s_state.colour);
	s_state.have_history = true;

	// value * 255 + 0.5, the output is now in history
	const float* result = s_state.history.data();
	for (u32 y = 0; y < height; y++)
	{
		u8* row = rgba + static_cast<size_t>(y) * stride;
		const float* in = result + static_cast<size_t>(y) * width * 3;
		for (u32 x = 0; x < width; x++)
		{
			for (u32 c = 0; c < 3; c++)
			{
				const float v = std::fmin(std::fmax(in[x * 3 + c], 0.0f), 1.0f);
				row[x * 4 + c] = static_cast<u8>(v * 255.0f + 0.5f);
			}
		}
	}

	return true;
}

static void WorkerThread()
{
	Threading::SetNameOfCurrentThread("DLSS-NR");

	// The model is opened and closed here, so libframe only ever sees this one thread.
	if (!OpenFrame())
	{
		s_worker.failed.store(true, std::memory_order_release);
		s_worker.busy.store(false, std::memory_order_release);
		return;
	}

	std::vector<u8> pixels;
	std::unique_lock lock(s_worker.mutex);
	for (;;)
	{
		s_worker.cv.wait(lock, [] { return s_worker.quit || s_worker.job_pending; });
		if (s_worker.quit)
			break;

		pixels.swap(s_worker.job);
		const u32 width = s_worker.job_width;
		const u32 height = s_worker.job_height;
		const float intensity = s_worker.job_intensity;
		s_worker.job_pending = false;
		lock.unlock();

		if (s_worker.reset_history.exchange(false, std::memory_order_acq_rel))
			s_state.have_history = false;

		const bool ok = ProcessFrame(pixels.data(), width, height, intensity);

		lock.lock();
		if (ok)
		{
			s_worker.result.swap(pixels);
			s_worker.result_width = width;
			s_worker.result_height = height;
			s_worker.result_ready = true;
		}
		s_worker.busy.store(false, std::memory_order_release);
	}
	lock.unlock();

	CloseFrame();
}

bool GSDLSSNR::IsBusy()
{
	return s_worker.busy.load(std::memory_order_acquire);
}

bool GSDLSSNR::Submit(std::vector<u8>& rgba, u32 width, u32 height, float intensity)
{
	if (width == 0 || height == 0 || s_worker.failed.load(std::memory_order_acquire) ||
		s_worker.busy.load(std::memory_order_acquire))
	{
		return false;
	}

	{
		std::unique_lock lock(s_worker.mutex);
		s_worker.job.swap(rgba);
		s_worker.job_width = width;
		s_worker.job_height = height;
		s_worker.job_intensity = intensity;
		s_worker.job_pending = true;
		s_worker.busy.store(true, std::memory_order_release);
	}

	if (!s_worker.thread.joinable())
	{
		s_worker.quit = false;
		s_worker.thread = std::thread(WorkerThread);
	}
	else
	{
		s_worker.cv.notify_one();
	}

	return true;
}

bool GSDLSSNR::Receive(std::vector<u8>& rgba, u32* width, u32* height)
{
	std::unique_lock lock(s_worker.mutex);
	if (!s_worker.result_ready)
		return false;

	rgba.swap(s_worker.result);
	*width = s_worker.result_width;
	*height = s_worker.result_height;
	s_worker.result_ready = false;
	return true;
}

void GSDLSSNR::ResetHistory()
{
	s_worker.reset_history.store(true, std::memory_order_release);
}

void GSDLSSNR::Shutdown()
{
	if (s_worker.thread.joinable())
	{
		{
			std::unique_lock lock(s_worker.mutex);
			s_worker.quit = true;
		}
		s_worker.cv.notify_one();
		s_worker.thread.join();
	}

	std::unique_lock lock(s_worker.mutex);
	s_worker.quit = false;
	s_worker.job_pending = false;
	s_worker.result_ready = false;
	s_worker.busy.store(false, std::memory_order_release);
	s_worker.reset_history.store(false, std::memory_order_release);
	// failed is kept: a model that couldn't open once won't open on the next device either.
}

#else

bool GSDLSSNR::IsAvailable()
{
	return false;
}

bool GSDLSSNR::IsBusy()
{
	return false;
}

bool GSDLSSNR::Submit(std::vector<u8>& rgba, u32 width, u32 height, float intensity)
{
	return false;
}

bool GSDLSSNR::Receive(std::vector<u8>& rgba, u32* width, u32* height)
{
	return false;
}

void GSDLSSNR::ResetHistory()
{
}

void GSDLSSNR::Shutdown()
{
}

#endif
