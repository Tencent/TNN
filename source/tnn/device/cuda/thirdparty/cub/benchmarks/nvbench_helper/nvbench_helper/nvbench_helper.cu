#include <cub/device/device_copy.cuh>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <random>
#include <type_traits>

#include "thrust/device_vector.h"
#include <curand.h>
#include <nvbench_helper.cuh>

namespace
{

constexpr double lognormal_mean  = 3.0;
constexpr double lognormal_sigma = 1.2;

enum class executor
{
  host,
  device
};

class host_generator_t
{
public:
  template <typename T>
  void generate(seed_t seed, cuda::std::span<T> device_span, bit_entropy entropy, T min, T max);

  const double* new_uniform_distribution(seed_t seed, std::size_t num_items);
  const double* new_lognormal_distribution(seed_t seed, std::size_t num_items);
  const double* new_constant(std::size_t num_items, double val);

private:
  thrust::host_vector<double> m_distribution;
};

const double* host_generator_t::new_uniform_distribution(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);
  double* h_distribution = thrust::raw_pointer_cast(m_distribution.data());

  std::default_random_engine re(seed.get());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (std::size_t i = 0; i < num_items; i++)
  {
    h_distribution[i] = dist(re);
  }

  return h_distribution;
}

const double* host_generator_t::new_lognormal_distribution(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);
  double* h_distribution = thrust::raw_pointer_cast(m_distribution.data());

  std::default_random_engine re(seed.get());
  std::lognormal_distribution<double> dist(lognormal_mean, lognormal_sigma);

  for (std::size_t i = 0; i < num_items; i++)
  {
    h_distribution[i] = dist(re);
  }

  return h_distribution;
}

const double* host_generator_t::new_constant(std::size_t num_items, double val)
{
  m_distribution.resize(num_items);
  double* h_distribution = thrust::raw_pointer_cast(m_distribution.data());
  thrust::fill_n(thrust::host, h_distribution, num_items, val);
  return h_distribution;
}

class device_generator_t
{
public:
  device_generator_t()
  {
    curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
  }

  ~device_generator_t()
  {
    curandDestroyGenerator(m_gen);
  }

  template <typename T>
  void generate(seed_t seed, cuda::std::span<T> device_span, bit_entropy entropy, T min, T max);

  const double* new_uniform_distribution(seed_t seed, std::size_t num_items);
  const double* new_lognormal_distribution(seed_t seed, std::size_t num_items);
  const double* new_constant(std::size_t num_items, double val);

private:
  curandGenerator_t m_gen;
  thrust::device_vector<double> m_distribution;
};

template <typename T>
struct random_to_item_t
{
  double m_min;
  double m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<double>(min))
      , m_max(static_cast<double>(max))
  {}

  __host__ __device__ T operator()(double random_value) const
  {
    if constexpr (std::is_floating_point_v<T>)
    {
      return static_cast<T>((m_max - m_min) * random_value + m_min);
    }
    else
    {
      return static_cast<T>(floor((m_max - m_min + 1) * random_value + m_min));
    }
  }
};

const double* device_generator_t::new_uniform_distribution(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);
  double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());

  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());
  curandGenerateUniformDouble(m_gen, d_distribution, num_items);

  return d_distribution;
}

const double* device_generator_t::new_lognormal_distribution(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);
  double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());

  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());
  curandGenerateLogNormalDouble(m_gen, d_distribution, num_items, lognormal_mean, lognormal_sigma);

  return d_distribution;
}

const double* device_generator_t::new_constant(std::size_t num_items, double val)
{
  m_distribution.resize(num_items);
  double* d_distribution = thrust::raw_pointer_cast(m_distribution.data());
  thrust::fill_n(thrust::device, d_distribution, num_items, val);
  return d_distribution;
}

struct and_t
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    return a & b;
  }

  __host__ __device__ float operator()(float a, float b) const
  {
    const std::uint32_t result = reinterpret_cast<std::uint32_t&>(a) & reinterpret_cast<std::uint32_t&>(b);
    return reinterpret_cast<const float&>(result);
  }

  __host__ __device__ double operator()(double a, double b) const
  {
    const std::uint64_t result = reinterpret_cast<std::uint64_t&>(a) & reinterpret_cast<std::uint64_t&>(b);
    return reinterpret_cast<const double&>(result);
  }

  __host__ __device__ complex operator()(complex a, complex b) const
  {
    double a_real = a.real();
    double a_imag = a.imag();

    double b_real = b.real();
    double b_imag = b.imag();

    const std::uint64_t result_real =
      reinterpret_cast<std::uint64_t&>(a_real) & reinterpret_cast<std::uint64_t&>(b_real);

    const std::uint64_t result_imag =
      reinterpret_cast<std::uint64_t&>(a_imag) & reinterpret_cast<std::uint64_t&>(b_imag);

    return {static_cast<float>(reinterpret_cast<const double&>(result_real)),
            static_cast<float>(reinterpret_cast<const double&>(result_imag))};
  }
};

struct set_real_t
{
  complex m_min{};
  complex m_max{};
  complex* m_d_in{};
  const double* m_d_tmp{};

  __host__ __device__ void operator()(std::size_t i) const
  {
    m_d_in[i].real(random_to_item_t<double>{m_min.real(), m_max.real()}(m_d_tmp[i]));
  }
};

struct set_imag_t
{
  complex m_min{};
  complex m_max{};
  complex* m_d_in{};
  const double* m_d_tmp{};

  __host__ __device__ void operator()(std::size_t i) const
  {
    m_d_in[i].imag(random_to_item_t<double>{m_min.imag(), m_max.imag()}(m_d_tmp[i]));
  }
};

template <class T>
struct lognormal_transformer_t
{
  std::size_t total_elements;
  double sum;

  __host__ __device__ T operator()(double val) const
  {
    return floor(val * total_elements / sum);
  }
};

class generator_t
{
public:
  template <typename T>
  void generate(executor exec, seed_t seed, cuda::std::span<T> span, bit_entropy entropy, T min, T max)
  {
    construct_guard(exec);

    if (exec == executor::device)
    {
      this->generate(thrust::device, *m_device_generator, seed, span, entropy, min, max);
    }
    else
    {
      this->generate(thrust::host, *m_host_generator, seed, span, entropy, min, max);
    }
  }

  template <typename T>
  void power_law_segment_offsets(executor exec, seed_t seed, cuda::std::span<T> span, std::size_t total_elements)
  {
    construct_guard(exec);

    if (exec == executor::device)
    {
      this->power_law_segment_offsets(thrust::device, *m_device_generator, seed, span, total_elements);
    }
    else
    {
      this->power_law_segment_offsets(thrust::host, *m_host_generator, seed, span, total_elements);
    }
  }

private:
  void construct_guard(executor exec)
  {
    if (exec == executor::device)
    {
      if (!m_device_generator)
      {
        m_device_generator.emplace();
      }
    }
    else
    {
      if (!m_host_generator)
      {
        m_host_generator.emplace();
      }
    }
  }

  template <typename ExecT, typename DistT, typename T>
  void generate(const ExecT& exec, DistT& dist, seed_t seed, cuda::std::span<T> span, bit_entropy entropy, T min, T max);

  template <typename ExecT, typename DistT>
  void generate(const ExecT& exec,
                DistT& dist,
                seed_t seed,
                cuda::std::span<complex> span,
                bit_entropy entropy,
                complex min,
                complex max);

  template <typename ExecT, typename DistT>
  void generate(
    const ExecT& exec, DistT& dist, seed_t seed, cuda::std::span<bool> span, bit_entropy entropy, bool min, bool max);

  template <typename ExecT, typename DistT, typename T>
  void power_law_segment_offsets(
    const ExecT& exec, DistT& dist, seed_t seed, cuda::std::span<T> span, std::size_t total_elements);

  std::optional<host_generator_t> m_host_generator;
  std::optional<device_generator_t> m_device_generator;
};

template <typename ExecT, typename DistT, typename T>
void generator_t::generate(
  const ExecT& exec, DistT& dist, seed_t seed, cuda::std::span<T> span, bit_entropy entropy, T min, T max)
{
  switch (entropy)
  {
    case bit_entropy::_1_000: {
      const double* uniform_distribution = dist.new_uniform_distribution(seed, span.size());

      thrust::transform(
        exec, uniform_distribution, uniform_distribution + span.size(), span.data(), random_to_item_t<T>(min, max));
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      T random_value = random_to_item_t<T>(min, max)(dist(rng));
      thrust::fill(exec, span.data(), span.data() + span.size(), random_value);
      return;
    }
    default: {
      const double* uniform_distribution = dist.new_uniform_distribution(seed, span.size());
      ++seed;

      thrust::transform(
        exec, uniform_distribution, uniform_distribution + span.size(), span.data(), random_to_item_t<T>(min, max));

      const int number_of_steps = static_cast<int>(entropy);

      constexpr bool is_device = std::is_same_v<DistT, device_generator_t>;
      using vec_t              = std::conditional_t<is_device, thrust::device_vector<T>, thrust::host_vector<T>>;
      vec_t tmp_vec(span.size());
      cuda::std::span<T> tmp(thrust::raw_pointer_cast(tmp_vec.data()), tmp_vec.size());

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        this->generate(is_device ? executor::device : executor::host, seed, tmp, bit_entropy::_1_000, min, max);

        thrust::transform(exec, span.data(), span.data() + span.size(), tmp.data(), span.data(), and_t{});
      }
      return;
    }
  };
}

template <typename ExecT, typename DistT>
void generator_t::generate(
  const ExecT& exec,
  DistT& dist,
  seed_t seed,
  cuda::std::span<complex> span,
  bit_entropy entropy,
  complex min,
  complex max)
{
  switch (entropy)
  {
    case bit_entropy::_1_000: {
      const double* uniform_distribution = dist.new_uniform_distribution(seed, span.size());
      thrust::for_each_n(
        exec, thrust::make_counting_iterator(0), span.size(), set_real_t{min, max, span.data(), uniform_distribution});
      ++seed;

      uniform_distribution = dist.new_uniform_distribution(seed, span.size());
      thrust::for_each_n(
        exec, thrust::make_counting_iterator(0), span.size(), set_imag_t{min, max, span.data(), uniform_distribution});
      ++seed;
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<double> dist(0.0f, 1.0f);
      const float random_imag = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      const float random_real = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      thrust::fill(exec, span.data(), span.data() + span.size(), complex{random_real, random_imag});
      return;
    }
    default: {
      const double* uniform_distribution = dist.new_uniform_distribution(seed, span.size());
      thrust::for_each_n(
        exec, thrust::make_counting_iterator(0), span.size(), set_real_t{min, max, span.data(), uniform_distribution});
      ++seed;

      uniform_distribution = dist.new_uniform_distribution(seed, span.size());
      thrust::for_each_n(
        exec, thrust::make_counting_iterator(0), span.size(), set_imag_t{min, max, span.data(), uniform_distribution});
      ++seed;

      const int number_of_steps = static_cast<int>(entropy);

      constexpr bool is_device = std::is_same_v<DistT, device_generator_t>;
      using vec_t = std::conditional_t<is_device, thrust::device_vector<complex>, thrust::host_vector<complex>>;

      vec_t tmp_vec(span.size());
      cuda::std::span<complex> tmp(thrust::raw_pointer_cast(tmp_vec.data()), tmp_vec.size());

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        this->generate(is_device ? executor::device : executor::host, seed, tmp, bit_entropy::_1_000, min, max);

        thrust::transform(exec, span.data(), span.data() + span.size(), tmp.data(), span.data(), and_t{}); // TODO issue
      }
      return;
    }
  };
}

struct random_to_probability_t
{
  double m_probability;

  __host__ __device__ bool operator()(double random_value) const
  {
    return random_value < m_probability;
  }
};

template <typename ExecT, typename DistT>
void generator_t::generate(
  const ExecT& exec,
  DistT& dist,
  seed_t seed,
  cuda::std::span<bool> span,
  bit_entropy entropy,
  bool /* min */,
  bool /* max */)
{
  if (entropy == bit_entropy::_0_000)
  {
    thrust::fill(exec, span.data(), span.data() + span.size(), false);
  }
  else if (entropy == bit_entropy::_1_000)
  {
    thrust::fill(exec, span.data(), span.data() + span.size(), true);
  }
  else
  {
    const double* uniform_distribution = dist.new_uniform_distribution(seed, span.size());

    thrust::transform(
      exec,
      uniform_distribution,
      uniform_distribution + span.size(),
      span.data(),
      random_to_probability_t{entropy_to_probability(entropy)});
  }
}

template <class T>
struct lognormal_adjust_t
{
  T* segment_sizes{};

  __host__ __device__ T operator()(std::size_t sid) const
  {
    return segment_sizes[sid] + 1;
  }
};

template <typename ExecT, typename DistT, typename T>
void generator_t::power_law_segment_offsets(
  const ExecT& exec, DistT& dist, seed_t seed, cuda::std::span<T> device_segment_offsets, std::size_t total_elements)
{
  const std::size_t total_segments   = device_segment_offsets.size() - 1;
  const double* uniform_distribution = dist.new_lognormal_distribution(seed, total_segments);

  if (thrust::count(exec, uniform_distribution, uniform_distribution + total_segments, 0.0) == total_segments)
  {
    uniform_distribution = dist.new_constant(total_segments, 1.0);
  }

  const double sum = thrust::reduce(exec, uniform_distribution, uniform_distribution + total_segments);

  thrust::transform(
    exec,
    uniform_distribution,
    uniform_distribution + total_segments,
    device_segment_offsets.data(),
    lognormal_transformer_t<T>{total_elements, sum});

  const int diff =
    total_elements
    - thrust::reduce(exec, device_segment_offsets.data(), device_segment_offsets.data() + device_segment_offsets.size());

  if (diff > 0)
  {
    thrust::tabulate(exec,
                     device_segment_offsets.data(),
                     device_segment_offsets.data() + diff,
                     lognormal_adjust_t<T>{device_segment_offsets.data()});
  }

  thrust::exclusive_scan(
    exec,
    device_segment_offsets.data(),
    device_segment_offsets.data() + device_segment_offsets.size(),
    device_segment_offsets.data());
}

template <typename T>
void gen(executor exec, seed_t seed, cuda::std::span<T> span, bit_entropy entropy, T min, T max)
{
  generator_t{}.generate(exec, seed, span, entropy, min, max);
}

} // namespace

namespace detail
{

template <typename T>
void gen_host(seed_t seed, cuda::std::span<T> span, bit_entropy entropy, T min, T max)
{
  gen(executor::host, seed, span, entropy, min, max);
}

template <typename T>
void gen_device(seed_t seed, cuda::std::span<T> device_span, bit_entropy entropy, T min, T max)
{
  gen(executor::device, seed, device_span, entropy, min, max);
}

template <class T>
struct offset_to_iterator_t
{
  T* base_it;

  __host__ __device__ __forceinline__ T* operator()(std::size_t offset) const
  {
    return base_it + offset;
  }
};

template <class T>
struct repeat_index_t
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
  {
    return thrust::constant_iterator<T>(static_cast<T>(i));
  }
};

struct offset_to_size_t
{
  std::size_t* offsets = nullptr;

  __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
  {
    return offsets[i + 1] - offsets[i];
  }
};

template <typename T>
void gen_key_segments(executor exec, seed_t seed, cuda::std::span<T> keys, cuda::std::span<std::size_t> segment_offsets)
{
  thrust::counting_iterator<int> iota(0);
  offset_to_iterator_t<T> dst_transform_op{keys.data()};

  const std::size_t total_segments = segment_offsets.size() - 1;

  auto d_range_srcs  = thrust::make_transform_iterator(iota, repeat_index_t<T>{});
  auto d_range_dsts  = thrust::make_transform_iterator(segment_offsets.data(), dst_transform_op);
  auto d_range_sizes = thrust::make_transform_iterator(iota, offset_to_size_t{segment_offsets.data()});

  if (exec == executor::device)
  {
    std::uint8_t* d_temp_storage   = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceCopy::Batched(
      d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceCopy::Batched(
      d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);
    cudaDeviceSynchronize();
  }
  else
  {
    for (std::size_t sid = 0; sid < total_segments; sid++)
    {
      thrust::copy(d_range_srcs[sid], d_range_srcs[sid] + d_range_sizes[sid], d_range_dsts[sid]);
    }
  }
}

template <class T>
struct ge_t
{
  T val;

  __host__ __device__ bool operator()(T x)
  {
    return x >= val;
  }
};

template <typename T>
std::size_t gen_uniform_offsets(
  executor exec,
  seed_t seed,
  cuda::std::span<T> segment_offsets,
  std::size_t min_segment_size,
  std::size_t max_segment_size)
{
  const T total_elements = segment_offsets.size() - 2;

  gen(exec,
      seed,
      segment_offsets,
      bit_entropy::_1_000,
      static_cast<T>(min_segment_size),
      static_cast<T>(max_segment_size));

  auto tail = [&](const auto& policy) {
    thrust::fill_n(policy, segment_offsets.data() + total_elements, 1, total_elements + 1);
    thrust::exclusive_scan(
      policy, segment_offsets.data(), segment_offsets.data() + segment_offsets.size(), segment_offsets.data());
    auto iter = thrust::find_if(
      policy, segment_offsets.data(), segment_offsets.data() + segment_offsets.size(), ge_t<T>{total_elements});
    auto dist = thrust::distance(segment_offsets.data(), iter);
    thrust::fill_n(policy, segment_offsets.data() + dist, 1, total_elements);
    return dist + 1;
  };

  if (exec == executor::device)
  {
    return tail(thrust::device);
  }

  return tail(thrust::host);
}

} // namespace detail

namespace detail
{

/**
 * @brief Generates a vector of random key segments.
 *
 * Not all parameter combinations can be satisfied. For instance, if the total
 * elements is less than the minimal segment size, the function will return a
 * vector with a single element that is outside of the requested range.
 * At most one segment can be out of the requested range.
 */
template <typename T>
void gen_uniform_key_segments_host(
  seed_t seed, cuda::std::span<T> keys, std::size_t min_segment_size, std::size_t max_segment_size)
{
  thrust::host_vector<std::size_t> segment_offsets(keys.size() + 2);

  {
    cuda::std::span<std::size_t> segment_offsets_span(
      thrust::raw_pointer_cast(segment_offsets.data()), segment_offsets.size());
    const std::size_t offsets_size =
      gen_uniform_offsets(executor::host, seed, segment_offsets_span, min_segment_size, max_segment_size);
    segment_offsets.resize(offsets_size);
  }

  cuda::std::span<std::size_t> segment_offsets_span(
    thrust::raw_pointer_cast(segment_offsets.data()), segment_offsets.size());

  gen_key_segments(executor::host, seed, keys, segment_offsets_span);
}

template <typename T>
void gen_uniform_key_segments_device(
  seed_t seed, cuda::std::span<T> keys, std::size_t min_segment_size, std::size_t max_segment_size)
{
  thrust::device_vector<std::size_t> segment_offsets(keys.size() + 2);

  {
    cuda::std::span<std::size_t> segment_offsets_span(
      thrust::raw_pointer_cast(segment_offsets.data()), segment_offsets.size());
    const std::size_t offsets_size =
      gen_uniform_offsets(executor::device, seed, segment_offsets_span, min_segment_size, max_segment_size);
    segment_offsets.resize(offsets_size);
  }

  cuda::std::span<std::size_t> segment_offsets_span(
    thrust::raw_pointer_cast(segment_offsets.data()), segment_offsets.size());

  gen_key_segments(executor::device, seed, keys, segment_offsets_span);
}

template <typename T>
std::size_t gen_uniform_segment_offsets_host(
  seed_t seed, cuda::std::span<T> segment_offsets, std::size_t min_segment_size, std::size_t max_segment_size)
{
  return gen_uniform_offsets(executor::host, seed, segment_offsets, min_segment_size, max_segment_size);
}

template <typename T>
std::size_t gen_uniform_segment_offsets_device(
  seed_t seed, cuda::std::span<T> segment_offsets, std::size_t min_segment_size, std::size_t max_segment_size)
{
  return gen_uniform_offsets(executor::device, seed, segment_offsets, min_segment_size, max_segment_size);
}

template <typename T>
void gen_power_law_segment_offsets_host(seed_t seed, cuda::std::span<T> segment_offsets, std::size_t elements)
{
  generator_t{}.power_law_segment_offsets<T>(executor::host, seed, segment_offsets, elements);
}

template <typename T>
void gen_power_law_segment_offsets_device(seed_t seed, cuda::std::span<T> segment_offsets, std::size_t elements)
{
  generator_t{}.power_law_segment_offsets<T>(executor::device, seed, segment_offsets, elements);
}

void do_not_optimize(const void* ptr)
{
  (void) ptr;
}

} // namespace detail

#define INSTANTIATE(TYPE)                                                                                       \
  template void detail::gen_power_law_segment_offsets_host<TYPE>(seed_t, cuda::std::span<TYPE>, std::size_t);   \
  template void detail::gen_power_law_segment_offsets_device<TYPE>(seed_t, cuda::std::span<TYPE>, std::size_t); \
  template std::size_t detail::gen_uniform_segment_offsets_host<TYPE>(                                          \
    seed_t, cuda::std::span<TYPE>, std::size_t, std::size_t);                                                   \
  template std::size_t detail::gen_uniform_segment_offsets_device<TYPE>(                                        \
    seed_t, cuda::std::span<TYPE>, std::size_t, std::size_t)

INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);

#undef INSTANTIATE

#define INSTANTIATE(TYPE)                                                                                               \
  template void detail::gen_uniform_key_segments_host<TYPE>(seed_t, cuda::std::span<TYPE>, std::size_t, std::size_t);   \
  template void detail::gen_uniform_key_segments_device<TYPE>(seed_t, cuda::std::span<TYPE>, std::size_t, std::size_t); \
  template void detail::gen_device<TYPE>(seed_t, cuda::std::span<TYPE>, bit_entropy, TYPE min, TYPE max);               \
  template void detail::gen_host<TYPE>(seed_t, cuda::std::span<TYPE>, bit_entropy, TYPE min, TYPE max)

INSTANTIATE(bool);

INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);

INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);

#if NVBENCH_HELPER_HAS_I128
INSTANTIATE(int128_t);
INSTANTIATE(uint128_t);
#endif

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(complex);
#undef INSTANTIATE
