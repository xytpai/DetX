#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


void assign_box_cuda(
	const at::Tensor &box, const int stride, 
	const float size_min, const float size_max, const float radius, 
	at::Tensor &target);
void assign_box(
	const at::Tensor &box, const int stride, 
	const float size_min, const float size_max, const float radius, 
	at::Tensor &target)
{
	CHECK_CUDA(box);
	CHECK_CUDA(target);
	assign_box_cuda(box, stride, size_min, size_max, radius, target);
}


at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
    const at::Tensor &targets, const int num_classes,
    const float gamma, const float alpha);
at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
    const at::Tensor &targets, const int num_classes, 
    const float gamma, const float alpha) {
  if (logits.type().is_cuda()) {
    return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma,
                                         alpha);
  }
}


at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
    const at::Tensor &targets, const at::Tensor &d_losses, 
    const int num_classes, const float gamma, const float alpha);
at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
    const at::Tensor &targets, const at::Tensor &d_losses,
    const int num_classes, const float gamma, const float alpha) {
  if (logits.type().is_cuda()) {
    return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses,
                                          num_classes, gamma, alpha);
  }
}


at::Tensor roi_align_corners_forward_cuda(const at::Tensor &features, const at::Tensor &rois, 
					const int feature_stride, const int out_size);
at::Tensor roi_align_corners_forward(const at::Tensor &features, const at::Tensor &rois, 
					const int feature_stride, const int out_size)
{
	CHECK_CUDA(features);
	CHECK_CUDA(rois);
	return roi_align_corners_forward_cuda(features, rois, feature_stride, out_size);
}


at::Tensor roi_align_corners_backward_cuda(
	const at::Tensor &d_losses, const at::Tensor &rois, 
	const int ph, const int pw, 
    const int feature_stride, const int out_size);
at::Tensor roi_align_corners_backward(
	const at::Tensor &d_losses, const at::Tensor &rois, 
	const int ph, const int pw, 
    const int feature_stride, const int out_size)
{
	CHECK_CUDA(d_losses);
	CHECK_CUDA(rois);
	return roi_align_corners_backward_cuda(d_losses, rois, ph, pw, feature_stride, out_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
	m.def("assign_box", &assign_box, "assign_box (CUDA)");
    m.def("sigmoid_focal_loss_forward", &SigmoidFocalLoss_forward,
        "sigmoid_focal_loss_forward (CUDA)");
    m.def("sigmoid_focal_loss_backward", &SigmoidFocalLoss_backward,
        "sigmoid_focal_loss_backward (CUDA)");
    m.def("roi_align_corners_forward", &roi_align_corners_forward, "roi_align_corners_forward (CUDA)");
	m.def("roi_align_corners_backward", &roi_align_corners_backward, "roi_align_corners_backward (CUDA)");
}
