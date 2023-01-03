import torch

from PIL import Image
from lucent.optvis import render, param, objectives
from lucent.modelzoo import inceptionv1
import torchvision.models
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
        param_f = lambda: param.image(128, fft=SPATIAL_DECORRELATION, decorrelate=CHANNEL_DECORRELATION, batch=2) # these motherfuckers coud have stated
        # that they use lambda with no arguments to declare that param_f, or image_f is a function. this is building a simple function, ffs.
       # opt = lambda params: torch.optim.Adam(params, 5e-2)
        #obj = "mixed3a:101"
        obj = objectives.channel("mixed3a", 230, batch=0) - objectives.channel("mixed3a", 222, batch=0)
        #obj = objectives.channel("mixed5a", 9) - 1e2*objectives.diversity("mixed5a")
        #obj = objectives.channel("mixed3a", 8)
        #obj = objectives.channel("mixed3a", 8) #+ objectives.blur_input_each_step()
        #weight = torch.rand(256, device=device)
        #obj = objectives.channel_weight("mixed3a", weight)
        #obj = objectives.channel("mixed3a", 101, batch=1) - objectives.channel("mixed3a", 101, batch=0)
        #direction = torch.rand(256, device=device)
        #obj = objectives.direction(layer='mixed3a', direction=direction)
        #Implementation with more than one objectives
        #channel = lambda n: objectives.channel("mixed4a", n)
        #obj = channel(476) + channel(465)
        
    #render.render_vis(model, obj, param_f)
    images_list = render.render_vis(model, obj, param_f)
    print(len(images_list))

if __name__ == "__main__":
    main()
