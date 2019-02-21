
#include "ATen/ATen.h"
#include "cub/cub.cuh"
#include "cuda_utils.h"

namespace FlexConvCuda {


constexpr __host__ __device__ int pmin(int x, int y) { return x <= y ? x : y; }

// Implementations
template <typename scalar_t, typename NBtype, int Dp = 3, int C_N = 256,
          int C_Dout = 32, int C_Din = 64>
struct ForwardKernel;

template <typename scalar_t, typename NBtype, int Dp, int C_N, int C_Dout,
          int C_Din>
__global__ void runForwardKernel(
    const ForwardKernel<scalar_t, NBtype, Dp, C_N, C_Dout, C_Din> kernel) {
  kernel();
}

template <typename scalar_t, typename NBtype, int Dp, int C_N, int C_Dout,
          int C_Din>
struct ForwardKernel {
  enum {
    PMIN = 3  // only for unrolling
  };

  void launch(int B) {
    dim3 block(C_N);
    dim3 grid((N - 1) / C_N + 1, (Dout - 1) / C_Dout + 1, B);

    size_t shm_size = (Dp + 1) * C_Din * C_Dout * sizeof(scalar_t);

    runForwardKernel<<<grid, block, shm_size>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    scalar_t* s_shm = DynamicSharedMemory<scalar_t>();

    scalar_t* s_theta = (scalar_t*)&s_shm[0];
    scalar_t* s_bias = (scalar_t*)&s_shm[Dp * C_Din * C_Dout];

    // glob ids
    int b = blockIdx.z;
    int n = blockIdx.x * C_N + threadIdx.x;

    scalar_t result[C_Dout];
    for (int dout = 0; dout < C_Dout; ++dout) {
      result[dout] = 0.0;
    }

    scalar_t p0[Dp];
#pragma unroll pmin(Dp, PMIN)
    for (int dp = 0; dp < Dp && n < N; ++dp) {
      p0[dp] = d_positions[b * Dp * N + dp * N + n];
    }

    for (int o_din = 0; o_din < Din; o_din += C_Din) {
      // load shm
      __syncthreads();
      for (int tid = threadIdx.x; tid < Dp * C_Din * C_Dout; tid += C_N) {
        int dp = tid / (C_Din * C_Dout);
        int din = (tid % (C_Din * C_Dout)) / C_Dout;
        int dout = tid % C_Dout;

        int g_dout = (dout + blockIdx.y * C_Dout);
        int g_din = o_din + din;

        if (g_dout < Dout && g_din < Din) {
          s_theta[dp * C_Din * C_Dout + din * C_Dout + dout] =
              d_theta[dp * Din * Dout + g_din * Dout + g_dout];

          if (!dp) s_bias[din * C_Dout + dout] = d_bias[g_din * Dout + g_dout];
        }
      }
      __syncthreads();

      if (n < N) {
        // Loop over K
        for (int k = 0; k < K && n < N; ++k) {
          NBtype nk = d_neighborhood[b * K * N + k * N + n];

          scalar_t q[Dp];
#pragma unroll pmin(Dp, PMIN)
          for (int dp = 0; dp < Dp; ++dp) {
            q[dp] = d_positions[b * Dp * N + dp * N + nk] - p0[dp];
          }

          // Loop over Din
          for (int din = 0; din < C_Din && (o_din + din) < Din; ++din) {
            scalar_t fk = d_features[b * Din * N + (o_din + din) * N + nk];

            // Loop over partial Dout
            for (int dout = 0;
                 dout < C_Dout && (dout + blockIdx.y * C_Dout) < Dout; ++dout) {
              scalar_t w = 0.0;

              for (int dp = 0; dp < Dp; ++dp)
                w += q[dp] * s_theta[dp * C_Din * C_Dout + din * C_Dout + dout];
              w += s_bias[din * C_Dout + dout];
              result[dout] += w * fk;
            }
          }
        }
      }
    }

    for (int dout = 0;
         dout < C_Dout && (dout + blockIdx.y * C_Dout) < Dout && n < N;
         ++dout) {
      d_output[b * Dout * N + (dout + blockIdx.y * C_Dout) * N + n] =
          result[dout];
    }
  }

  // features:     incoming features                          [B, Din, N].
  // position:     each datapoint in nd space                 [B, Dp, N].
  // neighborhood: all K nearest neighbors                	[B, K, N].
  const scalar_t* d_features;
  const scalar_t* d_positions;
  const NBtype* d_neighborhood;

  // theta:		parameters for kernel function             	[Dp,
  // Din, Dout]. bias:       	parameters for kernel function [Din, Dout].
  const scalar_t* d_theta;
  const scalar_t* d_bias;

  // output:       each feature description for each point   	[B, Dout, N].
  scalar_t* d_output;

  int N;
  int K;
  int Din;
  int Dout;
};

template <typename scalar_t>
struct BackwardThetaKernel;

template <typename scalar_t>
__global__ void runBackwardKernel(const BackwardThetaKernel<scalar_t> kernel) {
  kernel();
}

template <typename scalar_t>
struct BackwardThetaKernel {
  enum { C_N = 256, DP_MAX = 3, DEGREE_MAX = 2 };

  void launch() {
    dim3 block(C_N);
    dim3 grid(Dout, Din);

    runBackwardKernel<<<grid, block>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    typedef cub::BlockReduce<scalar_t, C_N> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    scalar_t theta_diff[DP_MAX];
    for (int dp = 0; dp < Dp; ++dp) theta_diff[dp] = 0;

    scalar_t bias_diff = 0;

    int dout = blockIdx.x;
    int din = blockIdx.y;

    for (int b = 0; b < B; ++b) {
      for (int n = threadIdx.x; n < N; n += C_N) {
        scalar_t topdiff = d_topdiff[b * Dout * N + dout * N + n];

        for (int k = 0; k < K; ++k) {
          int nk0 = d_neigh[b * N * K + 0 * N + n];
          int nk = d_neigh[b * N * K + k * N + n];

          scalar_t feature = d_features[b * Din * N + din * N + nk];
          for (int dp = 0; dp < Dp; ++dp) {
            scalar_t diffpos = d_pos[b * Dp * N + dp * N + nk] -
                            d_pos[b * Dp * N + dp * N + nk0];
            theta_diff[dp] += feature * diffpos * topdiff;
          }
          bias_diff += feature * topdiff;
        }
      }
    }

    for (int dp = 0; dp < Dp; ++dp) {
      // for (int dd = 0; dd < Ddegree; ++dd) {
      scalar_t thread_data = theta_diff[dp];
      scalar_t aggregate = BlockReduce(temp_storage).Sum(thread_data, N);

      if (!threadIdx.x) {
        d_theta_out[dp * Din * Dout + din * Dout + dout] = aggregate;
      }
      // }
      __syncthreads();
    }

    scalar_t thread_data = bias_diff;

    scalar_t aggregate = BlockReduce(temp_storage).Sum(thread_data, N);

    if (!threadIdx.x) d_bias_out[din * Dout + dout] = aggregate;
  }

  const scalar_t* d_topdiff;

  const scalar_t* d_pos;
  const scalar_t* d_features;
  const int* d_neigh;

  const scalar_t* d_theta;
  const scalar_t* d_bias;

  scalar_t* d_theta_out;
  scalar_t* d_bias_out;

  int B;
  int N;
  int K;
  int Ddegree;
  int Dp;
  int Din;
  int Dout;
};

template <typename scalar_t>
struct BackwardFeatureKernel;

template <typename scalar_t>
__global__ void runBackwardKernel(const BackwardFeatureKernel<scalar_t> kernel) {
  kernel();
}

template <typename scalar_t>
struct BackwardFeatureKernel {
  enum {
    C_N = 32,
    C_Dout = 32,  // multiple of Warpsize is better

    C_Din = 8  // reduce first
  };

  void launch(int B) {
    dim3 fblock(C_N, C_Din);
    dim3 fgrid((N - 1) / C_N + 1, (Din - 1) / C_Din + 1, B);

    const int theta_size = Dp * C_Din * C_Dout;
    const int bias_size = C_Din * C_Dout;
    const int topdiff_size = C_N * C_Dout;
    const int pos_size = C_N * K * Dp;
    const int nk_size = C_N * K;

    int shm =
        (theta_size + bias_size + topdiff_size + pos_size) * sizeof(scalar_t) +
        (nk_size) * sizeof(int);

    runBackwardKernel<<<fgrid, fblock, shm>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    // extern __shared__ float s_shm[];
    scalar_t* s_shm = DynamicSharedMemory<scalar_t>();

    int i_n = threadIdx.x;
    int i_din = threadIdx.y;

    int b = blockIdx.z;
    int n = blockIdx.x * C_N + i_n;
    int din = blockIdx.y * C_Din + i_din;

    scalar_t* s_theta = (scalar_t*)&s_shm[0];
    scalar_t* s_bias = (scalar_t*)&s_theta[Dp * C_Din * C_Dout];
    scalar_t* s_topdiff = (scalar_t*)&s_bias[C_Din * C_Dout];
    scalar_t* s_pos = (scalar_t*)&s_topdiff[C_N * C_Dout];
    int* s_nk = (int*)&s_pos[C_N * K * Dp];

    for (int k = threadIdx.y; k < K && n < N; k += blockDim.y) {
      int nk = d_neigh[b * K * N + k * N + n];
      s_nk[k * C_N + i_n] = nk;

      for (int i_dp = 0; i_dp < Dp; ++i_dp) {
        s_pos[k * C_N * Dp + i_dp * C_N + i_n] =
            d_pos[b * Dp * N + i_dp * N + nk];
      }
    }

    __syncthreads();

    for (int i_dp = 0; i_dp < Dp; ++i_dp) {
      scalar_t val0 = s_pos[0 * C_N * Dp + i_dp * C_N + i_n];
      __syncthreads();
      for (int k = threadIdx.y; k < K && n < N; k += blockDim.y) {
        s_pos[k * C_N * Dp + i_dp * C_N + i_n] -= val0;
      }
    }

    for (int dout_outer = 0; dout_outer < (Dout - 1) / C_Dout + 1;
         ++dout_outer) {
      __syncthreads();

      // fill s_theta
      int dout = dout_outer * C_Dout + i_n;
      if (din < Din && dout < Dout) {
        for (int i_dp = 0; i_dp < Dp; ++i_dp)
          s_theta[i_dp * C_Din * C_Dout + i_din * C_Dout + i_n] =
              d_theta[i_dp * Din * Dout + din * Dout + dout];

        s_bias[i_din * C_Dout + i_n] = d_bias[din * Dout + dout];
      }

      if (n < N) {
        for (int i_dout = threadIdx.y;
             i_dout < C_Dout && (dout_outer * C_Dout + i_dout) < Dout;
             i_dout += blockDim.y)
          s_topdiff[i_dout * C_N + i_n] =
              d_topdiff[b * Dout * N + (dout_outer * C_Dout + i_dout) * N + n];
      }

      for (int dout_inner = 0;
           dout_inner < C_Dout && (dout_outer * C_Dout + dout_inner) < Dout;
           ++dout_inner) {
        for (int k = 0; k < K; k++) {
          __syncthreads();

          if (n < N && din < Din) {
            scalar_t W = 0;
            for (int dp = 0; dp < Dp; ++dp) {
              const scalar_t diffpos = s_pos[k * C_N * Dp + dp * C_N + i_n];
              W += s_theta[dp * C_Din * C_Dout + i_din * C_Dout + dout_inner] *
                   diffpos;
            }
            W += s_bias[i_din * C_Dout + dout_inner];
            scalar_t value = W * s_topdiff[dout_inner * C_N + i_n];

            // atomicAdd(
            //     &d_features_out[b * Din * N + din * N + s_nk[k * C_N + i_n]],
            //     value);
            atomicAdd(
                &d_features_out[b * Din * N + din * N + s_nk[k * C_N + i_n]],
                value);
          }
        }
      }
    }
  }

  const scalar_t* d_topdiff;
  const scalar_t* d_pos;
  const scalar_t* d_features;
  const int* d_neigh;
  const scalar_t* d_theta;
  const scalar_t* d_bias;

  scalar_t* d_features_out;

  int N;
  int K;
  int Dp;
  int Din;
  int Dout;
};

}  // namespace FlexConvCuda

// Interface
template <class scalar_t, class NBtype>
struct ForwardKernelType {
  typedef FlexConvCuda::ForwardKernel<scalar_t, NBtype, 3, 128, 32, 32> type;
};

template <>
struct ForwardKernelType<float, int> {
  typedef FlexConvCuda::ForwardKernel<float, int, 3, 128, 32, 64> type;
};

void flex_conv_forward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood, 
    at::Tensor positions,
    at::Tensor output) 
{
    using NBtype = int;

    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);

    output.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_conv_forward_kernel_cpu", ([&]
    {
        using FKT = ForwardKernelType<scalar_t, NBtype>::type;

        FKT fwk;
        fwk.N = N;
        fwk.K = K;
        fwk.Din = Din;
        fwk.Dout = Dout;
        fwk.d_features = features.data<scalar_t>();
        fwk.d_positions = positions.data<scalar_t>();
        fwk.d_neighborhood = neighborhood.data<NBtype>();
        fwk.d_theta = theta.data<scalar_t>();
        fwk.d_bias = bias.data<scalar_t>();
        fwk.d_output = output.data<scalar_t>();

        fwk.launch(B);
    }));
}


void flex_conv_backward_kernel_cuda(
    at::Tensor features,
    at::Tensor theta,
    at::Tensor bias,
    at::Tensor neighborhood,
    at::Tensor positions,
    at::Tensor topdiff,
    at::Tensor grad_features,
    at::Tensor grad_theta, 
    at::Tensor grad_bias)
{
    using NBtype = int;

    // get dimensions
    const int B = neighborhood.size(0);
    const int K = neighborhood.size(1);
    const int N = neighborhood.size(2);
    const int Dp = theta.size(0);
    const int Din = theta.size(1);
    const int Dout = theta.size(2);

    grad_features.zero_();
    grad_theta.zero_();
    grad_bias.zero_();

    AT_DISPATCH_FLOATING_TYPES(
        features.type(), "flex_conv_backward_kernel_cpu", ([&] 
    {
        using BFK = FlexConvCuda::BackwardFeatureKernel<scalar_t>;

        BFK bfk;
        bfk.N = N;
        bfk.K = K;
        bfk.Dp = Dp;
        bfk.Din = Din;
        bfk.Dout = Dout;

        bfk.d_pos = positions.data<scalar_t>();
        bfk.d_neigh = neighborhood.data<NBtype>();
        bfk.d_features = features.data<scalar_t>();
        bfk.d_theta = theta.data<scalar_t>();
        bfk.d_bias = bias.data<scalar_t>();

        bfk.d_topdiff = topdiff.data<scalar_t>();

        bfk.d_features_out = grad_features.data<scalar_t>();

        bfk.launch(B);

        using BTK = FlexConvCuda::BackwardThetaKernel<scalar_t>;

        BTK btk;
        btk.B = B;
        btk.N = N;
        btk.K = K;
        btk.Dp = Dp;
        btk.Din = Din;
        btk.Dout = Dout;

        btk.d_pos = positions.data<scalar_t>();
        btk.d_neigh = neighborhood.data<NBtype>();
        btk.d_features = features.data<scalar_t>();
        btk.d_theta = theta.data<scalar_t>();
        btk.d_bias = bias.data<scalar_t>();

        btk.d_topdiff = topdiff.data<scalar_t>();

        btk.d_theta_out = grad_theta.data<scalar_t>();
        btk.d_bias_out = grad_bias.data<scalar_t>();

        btk.launch();
    }));
}
