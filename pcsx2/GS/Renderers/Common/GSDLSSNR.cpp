// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/Common/GSDLSSNR.h"

#include "common/Console.h"

#ifdef ENABLE_DLSSNR

#include "nr_frame.h"
extern "C" {
#include "nr_image.h"
}

#include "common/Threading.h"
#include "common/Image.h"

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace
{
	// The three stages of nr_frame_main.c, each on its own thread. With history the previous
	// output is what the next frame's features and composition need, so only one frame is in
	// the network or composition at a time and what overlaps is the next frame's conversion to
	// float; without it the stages each hold a frame.
	enum Stage : u32
	{
		STAGE_FEATURES, // RGBA8 -> colour, nr_frame_features_masked()
		STAGE_NETWORK, // nr_frame_run_features(), the head cropped
		STAGE_COMPOSE, // nr_frame_compose(), colour -> RGBA8
		STAGE_COUNT
	};

	// One frame's buffers, handed from stage to stage. There are as many as stages, so the
	// pipeline is full when every stage holds one.
	struct Packet
	{
		std::vector<u8> rgba; // the frame, filtered in place (alpha is kept)
		u32 width = 0;
		u32 height = 0;
		int network_width = 0;
		int network_height = 0;
		nr_frame_params params;
		bool want_history = false; // GSDLSSNR::Settings::history
		bool use_history = false; // the history and previous input below belong to this frame
		std::string control_mask; // GSDLSSNR::Settings::control_mask
		bool use_mask = false;

		std::vector<float> colour;
		std::vector<float> features; // (network_height, network_width, 16)
		std::vector<float> wide; // network head, (network_height, network_width, 4)
		std::vector<float> head; // the head cropped to (h, w, 4)
		std::vector<float> output;
		std::vector<float> mask; // the control mask at (h, w, 3), when use_mask
	};

	// The control mask, loaded and resized by the features stage only.
	struct ControlMask
	{
		std::string path;
		bool loaded = false;
		u32 source_width = 0;
		u32 source_height = 0;
		std::vector<float> source; // (source_height, source_width, 3)
		u32 width = 0;
		u32 height = 0;
		std::vector<float> resized; // (height, width, 3)
	};

	// Everything below is guarded by mutex, except the frame's own state inside libframe, which
	// the stages split between them (see FeaturesStage()).
	struct DLSSNRPipeline
	{
		std::array<std::thread, STAGE_COUNT> threads;
		std::mutex mutex;
		std::condition_variable cv;
		bool quit = false;
		bool opened = false;
		u32 running = 0; // FeaturesStage and ComposeStage threads still alive

		nr_frame* frame = nullptr;

		std::array<Packet, STAGE_COUNT> packets;
		std::vector<Packet*> free_packets;
		std::array<Packet*, STAGE_COUNT> queued{}; // waiting for a stage
		std::array<bool, STAGE_COUNT> active{}; // a stage is working on a packet

		// The extent the graph was last prepared for, only read and written by FeaturesStage.
		u32 prepared_width = 0;
		u32 prepared_height = 0;
		ControlMask control_mask; // FeaturesStage only

		// The temporal history: the previous output and the previous input (for the floor).
		// Written by ComposeStage, read by the frame after it, which FeaturesStage only lets
		// in once the composition is done, so the stages never touch them at the same time.
		std::vector<float> history;
		std::vector<float> previous;
		u32 history_width = 0;
		u32 history_height = 0;
		bool have_history = false;
		std::atomic_bool reset_history{false};

		bool result_ready = false;
		std::vector<u8> result;
		u32 result_width = 0;
		u32 result_height = 0;

		std::atomic_bool failed{false};
		std::atomic_bool error_reported{false};
	};

	static DLSSNRPipeline s_pipe;
} // namespace

bool GSDLSSNR::IsAvailable()
{
	return true;
}

// nr_frame_main.c's PROFILES, applied over nr_frame_defaults().
static void ApplyProfile(nr_frame_params* p, u32 profile)
{
	switch (profile)
	{
		case 1: // natural
			p->normalized_style = 1.0f / 128;
			p->local_tone = 1.0f;
			p->local_structure = 1.0f;
			break;
		case 2: // cinematic
			p->normalized_style = 2.0f / 128;
			p->local_tone = 1.0f;
			p->local_structure = 1.0f;
			break;
		case 3: // neutral
			p->normalized_style = 0.0f;
			p->local_tone = 0.0f;
			p->local_structure = 0.0f;
			break;
		case 4: // vendor
			p->normalized_style = 0.0f;
			p->local_tone = 1.0f;
			p->local_structure = 1.5f;
			break;
		case 0: // standard
		default:
			p->normalized_style = 0.0f;
			p->local_tone = 1.0f;
			p->local_structure = 1.0f;
			break;
	}
}

// nr_frame_main.c's main(): the defaults and the flags, then the profile, then the explicit
// overrides (`nr_frame.controls`).
static void BuildParams(nr_frame_params* p, const GSDLSSNR::Settings& settings)
{
	nr_frame_defaults(p);
	p->intensity = settings.intensity;
	p->detail_strength = settings.detail_strength;
	p->colour_strength = settings.colour_strength;
	p->detail_radius = settings.detail_radius;
	p->frame_index = settings.frame_index;
	ApplyProfile(p, settings.profile);
	if (settings.style_index >= 0)
		p->normalized_style = static_cast<float>(settings.style_index) / 128.0f;
	if (settings.local_tone >= 0.0f)
		p->local_tone = settings.local_tone;
	if (settings.local_structure >= 0.0f)
		p->local_structure = settings.local_structure;
	// --skin-structure / --auto-mask: either one turns the automatic mask on, the other left at -1.
	const bool have_skin = settings.skin_structure >= 0.0f;
	const bool have_auto = settings.auto_mask >= 0.0f;
	if (have_skin || have_auto)
	{
		p->automatic_mask = 1;
		p->skin_structure = have_skin ? settings.skin_structure : -1.0f;
		p->automatic_structure = have_auto ? settings.auto_mask : -1.0f;
	}
}

static void ReportError(const char* what)
{
	// libframe keeps one error string for every thread, so with the stages running at once
	// the message can belong to another stage's call. Needs proper testing.
	if (!s_pipe.error_reported.exchange(true, std::memory_order_acq_rel))
		Console.Error("DLSS-NR: %s() failed: %s", what, nr_frame_error());
}

// Called with the lock held: gives a packet that failed back, so Submit() can use it again.
static void DropPacket(Packet* packet)
{
	s_pipe.free_packets.push_back(packet);
}

// Waits for work at a stage. Returns nullptr when the pipeline is quitting.
static Packet* TakePacket(std::unique_lock<std::mutex>& lock, Stage stage)
{
	s_pipe.cv.wait(lock, [stage] { return s_pipe.quit || s_pipe.queued[stage]; });
	if (s_pipe.quit)
		return nullptr;

	Packet* packet = s_pipe.queued[stage];
	s_pipe.queued[stage] = nullptr;
	s_pipe.active[stage] = true;
	s_pipe.cv.notify_all(); // the previous stage may be waiting for this slot
	return packet;
}

// Hands a packet to the next stage, waiting for it to take the one before. False when quitting.
static bool PassPacket(std::unique_lock<std::mutex>& lock, Stage stage, Packet* packet)
{
	s_pipe.active[stage] = false;
	s_pipe.cv.wait(lock, [stage] { return s_pipe.quit || !s_pipe.queued[stage + 1]; });
	if (s_pipe.quit)
	{
		DropPacket(packet);
		return false;
	}

	s_pipe.queued[stage + 1] = packet;
	s_pipe.cv.notify_all();
	return true;
}

// Waits for the network thread to open the model. False if it failed or the pipeline quit.
static bool WaitForFrame(std::unique_lock<std::mutex>& lock)
{
	s_pipe.cv.wait(lock, [] { return s_pipe.quit || s_pipe.opened || s_pipe.failed.load(std::memory_order_acquire); });
	return !s_pipe.quit && s_pipe.opened;
}

static void ConvertColour(Packet* p)
{
	const size_t pixels = static_cast<size_t>(p->width) * p->height;
	nr_frame_geometry(static_cast<int>(p->height), static_cast<int>(p->width), &p->network_height, &p->network_width);
	p->colour.resize(pixels * 3);
	p->features.resize(static_cast<size_t>(p->network_height) * p->network_width * 16);

	// byte / 255, like image_io.py and nr_frame_main.c
	const u32 stride = p->width * 4;
	for (u32 y = 0; y < p->height; y++)
	{
		const u8* row = p->rgba.data() + static_cast<size_t>(y) * stride;
		float* out = p->colour.data() + static_cast<size_t>(y) * p->width * 3;
		for (u32 x = 0; x < p->width; x++)
		{
			out[x * 3 + 0] = static_cast<float>(row[x * 4 + 0]) * (1.0f / 255.0f);
			out[x * 3 + 1] = static_cast<float>(row[x * 4 + 1]) * (1.0f / 255.0f);
			out[x * 3 + 2] = static_cast<float>(row[x * 4 + 2]) * (1.0f / 255.0f);
		}
	}
}

// nr_frame_main.c's resize(): `nr_image.bilinear`, one axis at a time through nr_resize_axis,
// the axis plan mapping pixel centres onto the source.
static void ResizeBilinear(const float* source, u32 height, u32 width, u32 target_height, u32 target_width,
	std::vector<float>* output)
{
	constexpr u32 channels = 3;
	std::vector<float> current(source, source + static_cast<size_t>(height) * width * channels);
	std::vector<float> next;
	std::vector<int32_t> low, high;
	std::vector<float> weight;
	u32 h = height, w = width;
	for (int axis = 0; axis < 2; axis++)
	{
		const u32 extent = axis == 0 ? h : w;
		const u32 count = axis == 0 ? target_height : target_width;
		if (extent == count)
			continue;
		low.resize(count);
		high.resize(count);
		weight.resize(count);
		for (u32 i = 0; i < count; i++)
		{
			const float centre = (static_cast<float>(i) + 0.5f) * (static_cast<float>(extent) / static_cast<float>(count)) - 0.5f;
			const float floored = std::floor(centre);
			const int lo = static_cast<int>(floored < 0.0f ? 0.0f : floored > static_cast<float>(extent - 1) ? static_cast<float>(extent - 1) : floored);
			low[i] = lo;
			high[i] = lo + 1 > static_cast<int>(extent) - 1 ? static_cast<int>(extent) - 1 : lo + 1;
			const float frac = centre - static_cast<float>(lo);
			weight[i] = frac < 0.0f ? 0.0f : frac > 1.0f ? 1.0f : frac;
		}
		const u32 nh = axis == 0 ? count : h;
		const u32 nw = axis == 0 ? w : count;
		next.resize(static_cast<size_t>(nh) * nw * channels);
		nr_resize_axis(current.data(), static_cast<ptrdiff_t>(w) * channels, channels, 1, nh, nw, channels, axis,
			low.data(), high.data(), weight.data(), next.data());
		current.swap(next);
		h = nh;
		w = nw;
	}
	output->swap(current);
}

// Loads the control mask when its path changes, and resizes it when the frame's extent does.
// nr_frame refuses a mask of another shape; a game's frame changes size, so this resizes it
// the way nr_frame's --size resizes a picture. Needs proper testing.
static bool PrepareControlMask(Packet* p)
{
	ControlMask& mask = s_pipe.control_mask;
	if (p->control_mask != mask.path)
	{
		mask = {};
		mask.path = p->control_mask;
		if (!mask.path.empty())
		{
			RGBA8Image image;
			if (image.LoadFromFile(mask.path.c_str()) && image.GetWidth() > 0 && image.GetHeight() > 0)
			{
				// byte / 255 of R, G, B, as read_png() reduces any PNG to 8-bit RGB
				mask.source_width = image.GetWidth();
				mask.source_height = image.GetHeight();
				mask.source.resize(static_cast<size_t>(mask.source_width) * mask.source_height * 3);
				const u32* pixels = image.GetPixels();
				for (size_t i = 0; i < static_cast<size_t>(mask.source_width) * mask.source_height; i++)
				{
					for (u32 c = 0; c < 3; c++)
						mask.source[i * 3 + c] = static_cast<float>((pixels[i] >> (c * 8)) & 0xFF) / 255.0f;
				}
				mask.loaded = true;
				Console.WriteLn("DLSS-NR: control mask %s (%ux%u)", mask.path.c_str(), mask.source_width, mask.source_height);
			}
			else
			{
				Console.Error("DLSS-NR: could not load the control mask %s", mask.path.c_str());
			}
		}
	}

	if (!mask.loaded)
		return false;

	if (mask.width != p->width || mask.height != p->height)
	{
		ResizeBilinear(mask.source.data(), mask.source_height, mask.source_width, p->height, p->width, &mask.resized);
		mask.width = p->width;
		mask.height = p->height;
	}
	p->mask = mask.resized;
	return true;
}

static bool BuildFeatures(Packet* p)
{
	const float* history = p->use_history ? s_pipe.history.data() : nullptr;
	const float* mask = p->use_mask ? p->mask.data() : nullptr;
	if (nr_frame_features_masked(s_pipe.frame, p->colour.data(), static_cast<int>(p->height),
			static_cast<int>(p->width), history, mask, &p->params, p->features.data()) != 0)
	{
		ReportError("nr_frame_features_masked");
		return false;
	}
	return true;
}

static bool RunNetwork(Packet* p)
{
	const int W = p->network_width;
	p->wide.resize(static_cast<size_t>(p->network_height) * W * 4);
	p->head.resize(static_cast<size_t>(p->width) * p->height * 4);
	if (nr_frame_run_features(s_pipe.frame, p->features.data(), p->network_height, W, p->wide.data()) != 0)
	{
		ReportError("nr_frame_run_features");
		return false;
	}

	// geometry.crop(head)
	for (u32 y = 0; y < p->height; y++)
	{
		std::memcpy(p->head.data() + static_cast<size_t>(y) * p->width * 4,
			p->wide.data() + static_cast<size_t>(y) * W * 4, static_cast<size_t>(p->width) * 4 * sizeof(float));
	}
	return true;
}

static bool Compose(Packet* p)
{
	p->output.resize(static_cast<size_t>(p->width) * p->height * 3);
	const float* history = p->use_history ? s_pipe.history.data() : nullptr;
	const float* previous = p->use_history ? s_pipe.previous.data() : nullptr;
	const float* mask = p->use_mask ? p->mask.data() : nullptr;
	if (nr_frame_compose(s_pipe.frame, p->head.data(), p->colour.data(), static_cast<int>(p->height),
			static_cast<int>(p->width), history, previous, mask, &p->params, p->output.data()) != 0)
	{
		ReportError("nr_frame_compose");
		return false;
	}

	// value * 255 + 0.5
	const u32 stride = p->width * 4;
	for (u32 y = 0; y < p->height; y++)
	{
		u8* row = p->rgba.data() + static_cast<size_t>(y) * stride;
		const float* in = p->output.data() + static_cast<size_t>(y) * p->width * 3;
		for (u32 x = 0; x < p->width; x++)
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

// Inside libframe the stages touch different parts of the nr_frame (the features the index
// tables, noise and the graph's extent, the network the graph and its buffers, the composition
// its temporal gate table and detail split). What they share is the extent the graph is
// prepared for, which nr_frame_features_masked() changes, and with history the previous frame's
// output. So the conversion to float and the mask run straight away, and the features then wait
// until the network (and, with history, the composition) has nothing left.
static void FeaturesStage()
{
	Threading::SetNameOfCurrentThread("DLSS-NR Features");

	std::unique_lock lock(s_pipe.mutex);
	if (WaitForFrame(lock))
	{
		while (Packet* p = TakePacket(lock, STAGE_FEATURES))
		{
			lock.unlock();
			ConvertColour(p);
			p->use_mask = PrepareControlMask(p);
			lock.lock();

			const bool new_extent = p->width != s_pipe.prepared_width || p->height != s_pipe.prepared_height;
			const bool history = p->want_history;
			s_pipe.cv.wait(lock, [new_extent, history] {
				if (s_pipe.quit)
					return true;
				const bool network_idle = !s_pipe.queued[STAGE_NETWORK] && !s_pipe.active[STAGE_NETWORK];
				const bool compose_idle = !s_pipe.queued[STAGE_COMPOSE] && !s_pipe.active[STAGE_COMPOSE];
				return (!new_extent || network_idle) && (!history || (network_idle && compose_idle));
			});
			if (s_pipe.quit)
			{
				DropPacket(p);
				break;
			}
			s_pipe.prepared_width = p->width;
			s_pipe.prepared_height = p->height;

			if (s_pipe.reset_history.exchange(false, std::memory_order_acq_rel) ||
				p->width != s_pipe.history_width || p->height != s_pipe.history_height)
			{
				s_pipe.have_history = false;
			}
			p->use_history = history && s_pipe.have_history;
			lock.unlock();

			const bool ok = BuildFeatures(p);

			lock.lock();
			if (!ok)
			{
				s_pipe.active[STAGE_FEATURES] = false;
				// The graph may be half prepared, prepare it again next time.
				s_pipe.prepared_width = 0;
				s_pipe.prepared_height = 0;
				DropPacket(p);
				continue;
			}
			if (!PassPacket(lock, STAGE_FEATURES, p))
				break;
		}
	}

	s_pipe.active[STAGE_FEATURES] = false;
	s_pipe.running--;
	s_pipe.cv.notify_all();
}

static void ComposeStage()
{
	Threading::SetNameOfCurrentThread("DLSS-NR Compose");

	std::unique_lock lock(s_pipe.mutex);
	if (WaitForFrame(lock))
	{
		while (Packet* p = TakePacket(lock, STAGE_COMPOSE))
		{
			lock.unlock();
			const bool ok = Compose(p);
			lock.lock();

			if (ok)
			{
				// The display's buffer comes back as this packet's, for a later frame.
				s_pipe.result.swap(p->rgba);
				s_pipe.result_width = p->width;
				s_pipe.result_height = p->height;
				s_pipe.result_ready = true;

				// This output and input are the next frame's history and previous input. Without
				// history nothing reads them, and a frame turning history back on starts afresh.
				if (!p->want_history)
				{
					s_pipe.have_history = false;
				}
				else
				{
					s_pipe.history.swap(p->output);
					s_pipe.previous.swap(p->colour);
					s_pipe.history_width = p->width;
					s_pipe.history_height = p->height;
					s_pipe.have_history = true;
				}
			}
			else
			{
				s_pipe.have_history = false;
			}
			s_pipe.active[STAGE_COMPOSE] = false;
			DropPacket(p);
			s_pipe.cv.notify_all(); // the features stage waits for this frame
		}
	}

	s_pipe.active[STAGE_COMPOSE] = false;
	s_pipe.running--;
	s_pipe.cv.notify_all();
}

// Opens the model, runs the network, and closes the model once the other stages are gone,
// so the device is created and destroyed on the thread that submits to it.
static void NetworkStage()
{
	Threading::SetNameOfCurrentThread("DLSS-NR Network");

	// NULL: the weights compiled into libdlssnr.
	nr_frame* frame = nr_frame_open(nullptr);
	if (frame)
	{
		Console.WriteLn("DLSS-NR: running on %s (%s runtime, %s)", nr_frame_device(frame), nr_frame_runtime(),
			nr_frame_gemm_path(frame));
	}
	else
	{
		Console.Error("DLSS-NR: nr_frame_open() failed: %s", nr_frame_error());
	}

	std::unique_lock lock(s_pipe.mutex);
	s_pipe.frame = frame;
	s_pipe.opened = (frame != nullptr);
	if (!frame)
		s_pipe.failed.store(true, std::memory_order_release);
	s_pipe.cv.notify_all();

	if (frame)
	{
		while (Packet* p = TakePacket(lock, STAGE_NETWORK))
		{
			lock.unlock();
			const bool ok = RunNetwork(p);
			lock.lock();

			if (!ok)
			{
				s_pipe.active[STAGE_NETWORK] = false;
				DropPacket(p);
				s_pipe.cv.notify_all();
				continue;
			}
			if (!PassPacket(lock, STAGE_NETWORK, p))
				break;
		}
	}
	s_pipe.active[STAGE_NETWORK] = false;

	// The other stages may still be inside a libframe call on this frame.
	s_pipe.cv.wait(lock, [] { return s_pipe.running == 0; });
	s_pipe.frame = nullptr;
	s_pipe.opened = false;
	lock.unlock();

	if (frame)
	{
		nr_frame_close(frame);
		nr_frame_shutdown();
	}
}

bool GSDLSSNR::IsBusy()
{
	// Not started yet (or shut down): the next Submit() starts it.
	if (!s_pipe.threads[STAGE_NETWORK].joinable())
		return false;

	std::unique_lock lock(s_pipe.mutex);
	return s_pipe.free_packets.empty() || s_pipe.queued[STAGE_FEATURES];
}

bool GSDLSSNR::Submit(std::vector<u8>& rgba, u32 width, u32 height, const Settings& settings)
{
	if (width == 0 || height == 0 || s_pipe.failed.load(std::memory_order_acquire))
		return false;

	std::unique_lock lock(s_pipe.mutex);
	if (!s_pipe.threads[STAGE_NETWORK].joinable())
	{
		s_pipe.free_packets.clear();
		for (Packet& packet : s_pipe.packets)
			s_pipe.free_packets.push_back(&packet);
		s_pipe.queued = {};
		s_pipe.active = {};
		s_pipe.quit = false;
		s_pipe.opened = false;
		s_pipe.have_history = false;
		s_pipe.reset_history.store(false, std::memory_order_release);
		s_pipe.error_reported.store(false, std::memory_order_release);
		s_pipe.running = 2;
		s_pipe.threads[STAGE_NETWORK] = std::thread(NetworkStage);
		s_pipe.threads[STAGE_FEATURES] = std::thread(FeaturesStage);
		s_pipe.threads[STAGE_COMPOSE] = std::thread(ComposeStage);
	}

	if (s_pipe.free_packets.empty() || s_pipe.queued[STAGE_FEATURES])
		return false;

	Packet* p = s_pipe.free_packets.back();
	s_pipe.free_packets.pop_back();
	p->rgba.swap(rgba);
	p->width = width;
	p->height = height;

	BuildParams(&p->params, settings);
	p->want_history = settings.history;
	p->control_mask = settings.control_mask;

	s_pipe.queued[STAGE_FEATURES] = p;
	s_pipe.cv.notify_all();
	return true;
}

bool GSDLSSNR::Receive(std::vector<u8>& rgba, u32* width, u32* height)
{
	std::unique_lock lock(s_pipe.mutex);
	if (!s_pipe.result_ready)
		return false;

	rgba.swap(s_pipe.result);
	*width = s_pipe.result_width;
	*height = s_pipe.result_height;
	s_pipe.result_ready = false;
	return true;
}

void GSDLSSNR::ResetHistory()
{
	// Taken by the features stage before the next frame's features.
	s_pipe.reset_history.store(true, std::memory_order_release);
}

void GSDLSSNR::Shutdown()
{
	{
		std::unique_lock lock(s_pipe.mutex);
		s_pipe.quit = true;
	}
	s_pipe.cv.notify_all();
	for (std::thread& thread : s_pipe.threads)
	{
		if (thread.joinable())
			thread.join();
	}

	std::unique_lock lock(s_pipe.mutex);
	s_pipe.quit = false;
	s_pipe.queued = {};
	s_pipe.active = {};
	s_pipe.free_packets.clear();
	s_pipe.result_ready = false;
	s_pipe.have_history = false;
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

bool GSDLSSNR::Submit(std::vector<u8>& rgba, u32 width, u32 height, const Settings& settings)
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
