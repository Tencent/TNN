/*******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <memory>

#include "cub/detail/temporary_storage.cuh"
#include <c2h/catch2_test_helper.cuh>

using num_storage_slots = c2h::enum_type_list<int, 1, 4, 42>;

template <int Items>
std::size_t get_temporary_storage_size(std::size_t (&sizes)[Items])
{
  void* pointers[Items]{};
  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::AliasTemporaries(nullptr, temp_storage_bytes, pointers, sizes));
  return temp_storage_bytes;
}

std::size_t get_actual_zero()
{
  std::size_t sizes[1]{};

  return get_temporary_storage_size(sizes);
}

C2H_TEST("Test empty storage", "[temporary_storage_layout]", num_storage_slots)
{
  constexpr auto storage_slots = c2h::get<0, TestType>::value;
  cub::detail::temporary_storage::layout<storage_slots> temporary_storage;
  CHECK(temporary_storage.get_size() == get_actual_zero());
}

C2H_TEST("Test partially filled storage", "[temporary_storage_layout]", num_storage_slots)
{
  constexpr auto storage_slots             = c2h::get<0, TestType>::value;
  using target_type                        = std::uint64_t;
  constexpr std::size_t target_elements    = 42;
  constexpr std::size_t full_slot_elements = target_elements * sizeof(target_type);
  constexpr std::size_t empty_slot_elements{};

  cub::detail::temporary_storage::layout<storage_slots> temporary_storage;

  std::unique_ptr<cub::detail::temporary_storage::alias<target_type>> arrays[storage_slots];
  std::size_t sizes[storage_slots]{};

  for (int slot_id = 0; slot_id < storage_slots; slot_id++)
  {
    auto slot = temporary_storage.get_slot(slot_id);

    const std::size_t elements = slot_id % 2 == 0 ? full_slot_elements : empty_slot_elements;

    sizes[slot_id] = elements * sizeof(target_type);
    arrays[slot_id].reset(
      new cub::detail::temporary_storage::alias<target_type>(slot->template create_alias<target_type>(elements)));
  }

  const std::size_t temp_storage_bytes = temporary_storage.get_size();

  std::unique_ptr<std::uint8_t[]> temp_storage(new std::uint8_t[temp_storage_bytes]);

  temporary_storage.map_to_buffer(temp_storage.get(), temp_storage_bytes);

  CHECK(temp_storage_bytes == get_temporary_storage_size(sizes));

  for (int slot_id = 0; slot_id < storage_slots; slot_id++)
  {
    if (slot_id % 2 == 0)
    {
      CHECK(arrays[slot_id]->get() != nullptr);
    }
    else
    {
      CHECK(arrays[slot_id]->get() == nullptr);
    }
  }
}

C2H_TEST("Test grow", "[temporary_storage_layout]", num_storage_slots)
{
  constexpr auto StorageSlots                  = c2h::get<0, TestType>::value;
  using target_type                            = std::uint64_t;
  constexpr std::size_t target_elements_number = 42;

  cub::detail::temporary_storage::layout<StorageSlots> preset_layout;
  std::unique_ptr<cub::detail::temporary_storage::alias<target_type>> preset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    preset_arrays[slot_id].reset(new cub::detail::temporary_storage::alias<target_type>(
      preset_layout.get_slot(slot_id)->template create_alias<target_type>(target_elements_number)));
  }

  cub::detail::temporary_storage::layout<StorageSlots> postset_layout;
  std::unique_ptr<cub::detail::temporary_storage::alias<target_type>> postset_arrays[StorageSlots];

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    postset_arrays[slot_id].reset(new cub::detail::temporary_storage::alias<target_type>(
      postset_layout.get_slot(slot_id)->template create_alias<target_type>()));
    postset_arrays[slot_id]->grow(target_elements_number);
  }

  CHECK(preset_layout.get_size() == postset_layout.get_size());

  const std::size_t tmp_storage_bytes = preset_layout.get_size();
  std::unique_ptr<std::uint8_t[]> temp_storage(new std::uint8_t[tmp_storage_bytes]);

  preset_layout.map_to_buffer(temp_storage.get(), tmp_storage_bytes);
  postset_layout.map_to_buffer(temp_storage.get(), tmp_storage_bytes);

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    CHECK(postset_arrays[slot_id]->get() == preset_arrays[slot_id]->get());
  }
}

C2H_TEST("Test double grow", "[temporary_storage_layout]", num_storage_slots)
{
  constexpr auto storage_slots                 = c2h::get<0, TestType>::value;
  using target_type                            = std::uint64_t;
  constexpr std::size_t target_elements_number = 42;

  cub::detail::temporary_storage::layout<storage_slots> preset_layout;
  std::unique_ptr<cub::detail::temporary_storage::alias<target_type>> preset_arrays[storage_slots];

  for (int slot_id = 0; slot_id < storage_slots; slot_id++)
  {
    preset_arrays[slot_id].reset(new cub::detail::temporary_storage::alias<target_type>(
      preset_layout.get_slot(slot_id)->template create_alias<target_type>(2 * target_elements_number)));
  }

  cub::detail::temporary_storage::layout<storage_slots> postset_layout;
  std::unique_ptr<cub::detail::temporary_storage::alias<target_type>> postset_arrays[storage_slots];

  for (int slot_id = 0; slot_id < storage_slots; slot_id++)
  {
    postset_arrays[slot_id].reset(new cub::detail::temporary_storage::alias<target_type>(
      postset_layout.get_slot(slot_id)->template create_alias<target_type>(target_elements_number)));
    postset_arrays[slot_id]->grow(2 * target_elements_number);
  }

  CHECK(preset_layout.get_size() == postset_layout.get_size());

  const std::size_t tmp_storage_bytes = preset_layout.get_size();
  std::unique_ptr<std::uint8_t[]> temp_storage(new std::uint8_t[tmp_storage_bytes]);

  preset_layout.map_to_buffer(temp_storage.get(), tmp_storage_bytes);
  postset_layout.map_to_buffer(temp_storage.get(), tmp_storage_bytes);

  for (int slot_id = 0; slot_id < storage_slots; slot_id++)
  {
    CHECK(postset_arrays[slot_id]->get() == preset_arrays[slot_id]->get());
  }
}
