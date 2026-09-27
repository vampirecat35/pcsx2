// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "common/Pcsx2Defs.h"

#include <vector>

// DLSS-NR post-processing filter, through libframe (3rdparty/dlss-nr-on-vulkan, libdlssnr).
// One RGB frame in, one RGB frame out, the same way src/ref/nr_frame_main.c drives the library,
// with the previous output and input kept as the temporal history.
//
// The network runs on its own worker thread, one frame at a time, so the GS thread never waits
// for it: frames arriving while it is busy are skipped, and the display shows the newest result.
//
// Only available in builds configured with -DUSE_DLSSNR=ON, which link the model weights
// the owner supplies into the library. Needs proper testing.
namespace GSDLSSNR
{
	/// True when this build links libdlssnr.
	bool IsAvailable();

	/// True while the worker is filtering a frame (Submit() would refuse a new one).
	bool IsBusy();

	/// Hands a tightly packed RGBA8 image (width * 4 stride) to the worker, starting it on first use.
	/// The vector is swapped with a spare buffer, so its contents afterwards are unspecified.
	/// Returns false if the worker is busy or the filter failed; the vector is then untouched.
	bool Submit(std::vector<u8>& rgba, u32 width, u32 height, float intensity);

	/// Takes the newest filtered image, if one finished since the last call (alpha is kept).
	/// The vector is swapped with the result buffer.
	bool Receive(std::vector<u8>& rgba, u32* width, u32* height);

	/// Forgets the temporal history (e.g. after a resolution change or when toggled).
	void ResetHistory();

	/// Stops the worker, closes the model and releases its device.
	void Shutdown();
} // namespace GSDLSSNR
