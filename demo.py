import torch

from lucent.optvis import render, param, objectives
from lucent.modelzoo import inceptionv1

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1(pretrained=True)
    model.to(device).eval()

    CPPN = False

    SPATIAL_DECORRELATION = True #Change this to get either pixel image or fft (fourier transform on the image)
    CHANNEL_DECORRELATION = True #Change this to get decorellated colors or not 

    if CPPN:
        # CPPN parameterization
        param_f = lambda: param.cppn(224)
        opt = lambda params: torch.optim.Adam(params, 5e-3)
        # Some objectives work better with CPPN than others
        obj = "mixed4d_3x3_bottleneck_pre_relu_conv:139"
    else:
        param_f = lambda: param.image(224, channels=3, fft=SPATIAL_DECORRELATION, decorrelate=CHANNEL_DECORRELATION) # these motherfuckers coud have stated
        # that they use lambda with no arguments to declare that param_f, or image_f is a function. this is building a simple function, ffs.
        opt = lambda params: torch.optim.Adam(params, 5e-2)
        #obj = "mixed4a:476"
        obj = objectives.neuron("mixed4a", 476, 7, 7)

    render.render_vis(model, obj, param_f, opt)


if __name__ == "__main__":
    main()
