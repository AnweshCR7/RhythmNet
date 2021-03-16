import torch
import config as config

class MyLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss
        # return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output = torch.zeros(1).to(config.DEVICE)

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        # create a list of hr_outs without hr_t

        for hr in hr_outs:
            if hr == hr_t:
                pass
            else:
                output = output + (1/ctx.T)*torch.sign(ctx.hr_mean - hr)

        output = (1/ctx.T - 1)*torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None


# if __name__ == '__main__':
#
#     dtype = torch.float
#     device = torch.device("cpu")
#     # device = torch.device("cuda:0")  # Uncomment this to run on GPU
#     # torch.backends.cuda.matmul.allow_tf32 = False  # Uncomment this to run on GPU
#
#     # The above line disables TensorFloat32. This a feature that allows
#     # networks to run at a much faster speed while sacrificing precision.
#     # Although TensorFloat32 works well on most real models, for our toy model
#     # in this tutorial, the sacrificed precision causes convergence issue.
#     # For more information, see:
#     # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
#
#     # N is batch size; D_in is input dimension;
#     # H is hidden dimension; D_out is output dimension.
#     N, D_in, H, D_out = 64, 1000, 100, 10
#     # tensor([[0.4178, 0.8199, 0.1713, -0.8368, 0.2154, -0.4960, 0.4925, -0.7679,
#     #          -0.1096, 0.7345]], grad_fn= < SqueezeBackward1 >)
#     # Create random Tensors to hold input and outputs.
#     with torch.set_grad_enabled(True):
#         # hr_outs = torch.tensor([0.4178, 0.8199, 0.1713, -0.8368, 0.2154, -0.4960, 0.4925, -0.7679, -0.1096, 0.7345],
#         #                                                device=device, dtype=dtype)
#         hr_outs = torch.autograd.Variable(torch.randn(3), requires_grad=True)
#         hr_mean = hr_outs.mean()
#         # y = torch.tensor(0., device=device, dtype=dtype)
#
#         # Create random Tensors for weights.
#         # w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
#         # w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
#
#         learning_rate = 1e-6
#         smooth_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
#         for i in range(hr_outs.shape[0]):
#             # To apply our Function, we use Function.apply method. We alias this as 'relu'.
#             custom_loss = MyLoss.apply
#             smooth_loss = smooth_loss + custom_loss(hr_outs[i], hr_outs, hr_outs.shape[0])
#
#         smooth_loss.backward()
#
#     print("done")
