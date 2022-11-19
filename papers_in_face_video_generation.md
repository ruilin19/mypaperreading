# A Novice in face video generation
## Base generate model
[1]:https://arxiv.org/abs/1312.6114
[2]:https://zhuanlan.zhihu.com/p/144649293
[3]:https://export.arxiv.org/pdf/2006.11239.pdf
[4]:https://arxiv.org/pdf/2105.05233.pdf
[5]:https://export.arxiv.org/pdf/2209.02646v7.pdf
[6]:https://arxiv.org/pdf/1410.8516.pdf
[7]:http://export.arxiv.org/pdf/1605.08803
[8]:https://papers.nips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf
[9]:https://www.youtube.com/watch?v=uXY18nzdSsM&t=2983s
[10]:http://export.arxiv.org/pdf/2003.08934v1.pdf
[11]:https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
[12]:https://export.arxiv.org/pdf/1703.10593.pdf
[13]:https://arxiv.org/abs/1812.04948v1
[14]:https://arxiv.org/pdf/1912.04958.pdf
[15]:https://arxiv.org/pdf/2106.12423.pdf
[16]:https://mi.informatik.uni-siegen.de/projects_data/morphmod1.pdf
|method|source|
|:----:|:---|
|VAE|1. [Auto-Encoding Variational Bayes][1]<br>2. [半小时理解变分自编码器][2]|
|Diffusion Model|1. [Denoising Diffusion Probabilistic Models][3]<br>2. [Diffusion Models Beat GANs on Image Synthesis][4]<br>3. [A Survey on Generative Diffusion Model][5]|
|Glow|1. [NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION][6]<br>2. [DENSITY ESTIMATION USING REAL NVP][7]<br>3. [Glow: Generative Flow with Invertible 1×1 Convolutions][8]<br>4. [Flow-based Generative Model taught by Hung-yi Lee][9]|
|NeRf|1. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis][10]|
|GAN|1. [Generative Adversarial Nets][11]<br>2. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks][12]<br>3. [A Style-Based Generator Architecture for Generative Adversarial Networks][13]<br>4. [Analyzing and Improving the Image Quality of StyleGAN][14]<br>5. [Alias-Free Generative Adversarial Networks][15]|
|3DMM|1. [A Morphable Model For The Synthesis Of 3D Faces][16]|
## Some papers of face video generation
### Paper list
<span id="jump1"></span>

1. [Live Speech Portraits: Real-Time Photorealistic Talking-Head Animation][p1]
    + [code][code1]

<span id="jump2"></span>

2. [Photorealistic Audio-driven Video Portraits][p2]
    + [code][code2]

<span id="jump3"></span>

3. [Imitating Arbitrary Talking Style for Realistic Audio-Driven Talking Face Synthesis][p3]
    + [code][code3]

<span id="jump4"></span>

4. [FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning][p4]
    + [code][code4]

<span id="jump5"></span>

5. [LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces from Video using Pose and Lighting Normalization
][p5]

<span id="jump6"></span>

6. [AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis][p6]
    + [code][code6]

<span id="jump7"></span>

7. [DFA-NeRF: Personalized Talking Head Generation via Disentangled Face Attributes Neural Rendering][p7]

<span id="jump8"></span>

8. [Flow-guided One-shot Talking Face Generation with a High-resolution
Audio-visual Dataset][p8]
   + [code][code8]

<span id="jump9"></span>

9. [PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering][p9]
    + [code][code9]

<span id="jump10"></span>

10. [Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation][p10]
      + [code][code10]

<span id="jump11"></span>

11. [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pretrained StyleGAN][p11]

<span id="jump12"></span>

12. [Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion][p12]
       + [code][code12]

<span id="jump13"></span>

13. [One-shot Talking Face Generation from Single-speaker Audio-Visual Correlation Learning][p13]

<span id="jump14"></span>

14. [TRANSFORMER-S2A: ROBUST AND EFFICIENT SPEECH-TO-ANIMATION][p14]

<span id="jump15"></span>

15. [Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose][p15]
      + [code][code15]

<span id="jump16"></span>

16. [HeadGAN: One-shot Neural Head Synthesis and Editing][p16]

<span id="jump17"></span>

17. [Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation][p17]
     + [code][code17]


### Method Classification and Summary
**number is the papaer number in paper list**
|papers|methods|
|:---:|:---:|
|[11](#jump11) [16](#jump16) [13](#jump13)|pre-trained strategy|
|[13](#jump13) [14](#jump14) [7](#jump7)|Transformer|
|[17](#jump17)  [7](#jump7) [6](#jump6)|NeRF|
|[8](#jump8) [9](#jump9) [11](#jump11) [12](#jump12) [13](#jump13) [16](#jump16) |Dense Flow warp|
|[2](#jump2) [3](#jump3) [4](#jump4) [7](#jump7) [8](#jump8) [9](#jump9) [11](#jump11) [15](#jump15) [16](#jump16)|3DMM|




[p1]:https://arxiv.org/abs/2109.10595
[code1]:https://github.com/YuanxunLu/LiveSpeechPortraits
[p2]:https://pubmed.ncbi.nlm.nih.gov/32941145/
[code2]:https://github.com/xinwen-cs/AudioDVP
[p3]:https://export.arxiv.org/pdf/2111.00203.pdf
[code3]:https://github.com/wuhaozhe/style_avatar
[p4]:https://arxiv.org/abs/2108.07938
[code4]:https://github.com/zhangchenxu528/FACIAL
[p5]:https://openaccess.thecvf.com/content/CVPR2021/papers/Lahiri_LipSync3D_Data-Efficient_Learning_of_Personalized_3D_Talking_Faces_From_Video_CVPR_2021_paper.pdf
[p6]:https://arxiv.org/abs/2103.11078
[code6]:https://github.com/YudongGuo/AD-NeRF
[p7]:https://export.arxiv.org/pdf/2201.00791.pdf
[p8]:https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf
[code8]:https://github.com/MRzzm/HDTF
[p9]:https://arxiv.org/abs/2109.08379
[code9]:https://github.com/RenYurui/PIRender
[p10]:https://arxiv.org/abs/2104.11116
[code10]:https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS
[p11]:https://export.arxiv.org/pdf/2203.04036.pdf
[p12]:https://arxiv.org/pdf/2107.09293.pdf
[code12]:https://github.com/wangsuzhen/Audio2Head
[p13]:https://export.arxiv.org/pdf/2112.02749.pdf
[p14]:https://readpaper.com/pdf-annotate/note?pdfId=4667018670501806081&noteId=747432017719607296
[p15]:https://arxiv.org/abs/2002.10137
[code15]:https://github.com/yiranran/Audio-driven-TalkingFace-HeadPose
[p16]:https://arxiv.org/pdf/2012.08261.pdf
[p17]:https://export.arxiv.org/pdf/2201.07786.pdf
[code17]:https://github.com/alvinliu0/SSP-NeRF