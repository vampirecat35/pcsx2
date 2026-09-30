// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "common/Pcsx2Defs.h"

#include <string>
#include <vector>

// DLSS-NR post-processing filter, through libframe (3rdparty/dlss-nr-on-vulkan, libdlssnr).
// One RGB frame in, one RGB frame out, the same way src/ref/nr_frame_main.c drives the library:
// the same parameters (profile, then the explicit overrides), the optional control mask,
// features, the network at the vendor's extent, the head cropped, then the composition.
// Without history the output is nr_frame's for the same picture, byte for byte.
//
// With history the previous output (and, for the composition's floor, the previous input) goes
// into the features and the composition as well, which is the one place this differs.
//
// The three steps run on three threads (features, network, composition), so the GS thread never
// waits. With history a frame's features wait for the previous frame's composition, so only one
// frame is in the network or composition at a time; without it up to three frames are in
// flight. Frames arriving while the pipeline is full are skipped, and the display shows the
// newest result.
//
// Only available in builds configured with -DUSE_DLSSNR=ON, which link the model weights
// the owner supplies into the library. Needs proper testing.
namespace GSDLSSNR
{
	/// nr_frame's command line, as settings.
	struct Settings
	{
		float intensity = 1.0f; // --intensity
		u32 profile = 0; // --profile: standard, natural, cinematic, neutral, vendor
		int style_index = -1; // --style-index, negative keeps the profile's
		float local_tone = -1.0f; // --local-tone, negative keeps the profile's
		float local_structure = -1.0f; // --local-structure, negative keeps the profile's
		float skin_structure = -1.0f; // --skin-structure, negative leaves it unset
		float auto_mask = -1.0f; // --auto-mask, negative leaves it unset
		float detail_strength = 1.0f; // --detail-strength
		float colour_strength = 1.0f; // --colour-strength
		float detail_radius = 4.0f; // --detail-radius
		int frame_index = 0; // --frame-index
		std::string control_mask; // --control-mask, a PNG, resized to the frame
		bool history = true; // not nr_frame's: feed the previous output back in
	};

	/// True when this build links libdlssnr.
	bool IsAvailable();

	/// True while the pipeline can't take another frame (Submit() would refuse it).
	bool IsBusy();

	/// Hands a tightly packed RGBA8 image (width * 4 stride) to the pipeline, starting it on first use.
	/// The vector is swapped with a spare buffer, so its contents afterwards are unspecified.
	/// Returns false if the pipeline is full or the filter failed; the vector is then untouched.
	bool Submit(std::vector<u8>& rgba, u32 width, u32 height, const Settings& settings);

	/// Takes the newest filtered image, if one finished since the last call (alpha is kept).
	/// The vector is swapped with the result buffer.
	bool Receive(std::vector<u8>& rgba, u32* width, u32* height);

	/// Forgets the temporal history (e.g. after a resolution change or when toggled).
	void ResetHistory();

	/// Stops the pipeline, closes the model and releases its device.
	void Shutdown();

	/// Vulkan objects the renderer lends to the model, as opaque pointers so this header needs no
	/// Vulkan include: VkInstance, VkPhysicalDevice, VkDevice, VkQueue. The requirements are the
	/// ones documented for nr_frame_adopt_vulkan().
	struct VulkanShare
	{
		void* instance = nullptr;
		void* physical_device = nullptr;
		void* device = nullptr;
		// A compute-capable queue of queue_family. lock/unlock bracket every submit the model
		// makes; the renderer takes the same lock around its own submits, presents and idle waits.
		void* queue = nullptr;
		u32 queue_family = 0;
		bool cooperative_matrix = false; // VK_KHR_cooperative_matrix is enabled on device
		bool workgroup_memory_explicit_layout = false; // VK_KHR_workgroup_memory_explicit_layout too
		void* get_instance_proc_addr = nullptr; // the vkGetInstanceProcAddr that made instance
		void (*lock)(void*) = nullptr;
		void (*unlock)(void*) = nullptr;
		void* lock_context = nullptr;
	};

	/// Registers the device the model runs on from its next start. A running pipeline is stopped
	/// first, so the next Submit() reopens the model there. If the device won't take the model,
	/// it falls back to an instance of its own.
	void ShareVulkan(const VulkanShare& share);

	/// Stops the pipeline and drops the registered share. Call before destroying the device.
	void WithdrawVulkanShare();
} // namespace GSDLSSNR
