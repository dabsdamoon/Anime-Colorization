# Anime-Colorization (English)
Keras Implementation of different algorithms to color gray images of anime characters

## Data Used
### Source
One can obtain the dataset used from: https://www.kaggle.com/mylesoneill/tagged-anime-illustrations#danbooru-metadata.zip. 
Since danbooru image dataset is too big, only moeimouto-faces.zip dataset has been used

### Preprocessing
For better colorization algorithm, I've converted RGB image to LAB image and use L channel for input and AB channel as output.
The reason is that using L channel as input would let you keep general information of images as much as possible, whereas using onel fo RGB channel as input would exclude the information of two other channels. For more information on LAB image, you can go to the links below:

<p> (1) https://en.wikipedia.org/wiki/CIELAB_color_space</p>
<p> (2) https://www.aces.edu/dept/fisheries/education/pond_to_plate/documents/ExplanationoftheLABColorSpace.pdf</p>

Below diagram is data preprocessing process I take for the analysis:

![data_preprocessing](https://user-images.githubusercontent.com/43874313/49502850-5a27dc00-f8b9-11e8-9f91-b636b29d78eb.png)


## Algorithms Used (with Reference)
### Alpha Version algorithm by Emil Wallner 
<p>(1) https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/</p> 

This is a simple CNN encoder-decoder algorithm that Emil Wallner created to colorize the image. Roughly, the algorithm has a structure like a below diagram, and you can check a detailed Keras code on the website link above.
![alt text](https://blog.floydhub.com/content/images/2018/06/image_scaling_proces.png)


### U-Net Implementation
<p>(1) https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/</p>

According to the inventor of the algorithm, U-Net is originalled intended to be convolutional network architecture for fast and precise segmentation of images. While searching for colorization algorithms, however, I've seen quite many people using U-Net for colorization, and since U-Net also has a structure of eocnder-decoder format suitable for colorization, I've editted the U-Net Keras code (https://github.com/zhixuhao/unet) little bit so that it can be used for colorization.

Below diagram is an architecture of U-Net algorithm:

![alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

### Full Version by Emil Wallner (Alpha Version algorithm with Fusion Layer)
<p>(1) http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en//</p>

In Emil's post, he didn't end up implementing Alpha algorithm, but he furtuer improved it by applying a concept called 'Fusion Layer'. Basically, what fusion layer does (in my case of coloring anime faces) is that if an input comes into the algorithm, it adds information about which anime character the input is to the encoded vector. Then, the algorithm begins to color it. Simply saying, it is based on the assumption that classifying the input first would help the algorithm giving a better colorization result. If you are interested in the concept of the fusion layer, below is a diagram briefly showing the concept of fusion layer (For more details, visit the link above). 

![alt text](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/images/model.png)

Note: Emil used the vector output of InceptionV3 as a fusion layer, whereas I used the vector output of ResNet as a fusion layer.

### DCGAN
<p>(1) http://cs231n.stanford.edu/reports/2017/pdfs/302.pdf</p>
<p>(2) https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py</p>

Originally introduced by Ian Goodfellow in 2014, GAN is still popular deep-learning algorithm used for various purpuses. Recently, I've read a paper (1) that used DCGAN (GAN with CNN architecture) for image colorization, so I also decided to apply the algorithm for my anime face colorization. Python code for GAN I've wrote is originally from Erik Lindernoren's github (2).

![alt text](https://gluon.mxnet.io/_images/dcgan.png)
(Image from https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html)

### WGAN
<p>(1) https://arxiv.org/abs/1701.07875</p>
<p>(2) https://medium.com/@sunnerli/the-story-about-wgan-784be5acd84c</p>
<p>(3) https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i</p>
<p>(4) https://vincentherrmann.github.io/blog/wasserstein/(/p>

As GAN gets more popular as deep-learning algorithm, people have also been focusing on disadvantages of GAN. Thus, many new versions of GAN trying to remove those disadvantages have been invented. WGAN(Wasserstein GAN) is one of those new versions by Arjovsky and Bottou (2017)(1), which applies a concept of Wasserstein Distance (Earth Mover's Distance). Instead of KL and JS divergence used for distance as a loss function in original GAN, WGAN use Wasserstein distance as a new loss function. The very brief reason of changing the loss function is due to the inability of KL and JS divergence to correctly capture the loss value, whereas Wasserstein distance metric is able to capture the value. 

![alt text](https://cdn-images-1.medium.com/max/800/1*xRjphX2OGhfDllYFIkabzw.png "Brief Explanation of Divergence Metrics")

For more details about Wasserstein distance, visit great articles and presentations I listed above ((2), (3)), and I've referred to Vincent Herrmann's code (4) for Keras WGAN implementation. Later, I'll create another repo exclusively explaining WGAN as much as I can.

## Acknowledgement
Special thanks to Neowiz Play Studio (http://neowizplaystudio.com/ko/) and Sungkyu Oh (hanmaum@neowiz.com) who have not only answered the problems I've encountered but also provided up-to-date machines (including RTX 2080 ti!!) used for analysis.

## Contact Information
<p>facebook: https://www.facebook.com/dabin.moon.7 </p>
<p>email: dabsdamoon@neowiz.com</p>


# Anime-Colorization (Korean)
흑백 애니메이션 캐릭터 이미지를 색칠하는 keras 알고리즘을 작성하였습니다.  

## 사용 데이터
### 출처
알고리즘 생성에 사용된 데이터는 이 링크에서 획득하실 수 있습니다: https://www.kaggle.com/mylesoneill/tagged-anime-illustrations#danbooru-metadata.zip. 
danbooru image dataset은 너무 커서 moeimouto-faces.zip dataset 만 사용하였습니다.

### 데이터 가공
좀 더 나은 colorization algorithm을 위해서 RGB 이미지 대신 LAB형태의 이미지를 사용였고, 밝기를 상징하는 L channel을 input으로, 나머지 색을 관장하는 A,B channel을 output으로 도출하는 알고리즘으로 작성해보았습니다. LAB 이미지와 관련한 좀 더 자새한 정보는 아래 두 링크를 참조하시면 되겠습니다.

<p> (1) https://en.wikipedia.org/wiki/CIELAB_color_space</p>
<p> (2) https://www.aces.edu/dept/fisheries/education/pond_to_plate/documents/ExplanationoftheLABColorSpace.pdf</p>

아래는 제가 진행한 데이터 가공 과정을 간단하게 설명해주는 도표입니다:

![data_preprocessing](https://user-images.githubusercontent.com/43874313/49502850-5a27dc00-f8b9-11e8-9f91-b636b29d78eb.png)


## 사용 알고리즘 (with 참조 포함)
### Alpha Version algorithm by Emil Wallner 
<p>(1) https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/</p> 

Alpha 버전 알고리즘은 간단한 CNN encoder-decoder 형태의 알고리즘으로써 Emil Wallner 라는 분이 이미지 색칠을 위해 사용하신 알고리즘입니다. 알고리즘의 대강 구조는 아래 그림과 같고, 자세한 설명 및 코드는 위 링크를 참조하시면 되겠습니다. 

![alt text](https://blog.floydhub.com/content/images/2018/06/image_scaling_proces.png)


### U-Net Implementation
<p>(1) https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/</p>

U-Net은 원래 CNN 구조를 이용한 image segmentation이 주 목적인 알고리즘입니다. 하지만, colorization 알고리즘들을 공부하던 도중, 꽤 많은 사람들이 U-Net을 이용하여 colorization 알고리즘을 구현하는 모습을 보았고, 저 역시도 U-Net의 구조가 encoder-decoder 형태인지라 colorization 업무에 적합하다고 생각하여 U-Net을 적용해보게 되었습니다. 본 링크(https://github.com/zhixuhao/unet)의 Keras로 구현된 U-Net 코드를 참고하였습니다. 

아래의 그림은 U-Net의 구조를 보여주는 그림입니다:

![alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

### Full Version by Emil Wallner (Alpha Version algorithm with Fusion Layer)
<p>(1) http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en//</p>

위에서 언급한 Emil의 글을 읽어보면 Emil은 coloriztaion을 Alpha 알고리즘에서 끝내기 않고 더 나아가 Fusion Layer라는 컨셉을 적용하여 더욱 발전시킵니다. 여기서의 Fusion Layer는 (애니메이션 캐릭터 색칠이 주 목적일 경우) 간단하게 설명해서 input이 들어왔을 때, 이 input이 어떤 캐릭터인지 먼저 classify를 해주고, 그 다음에 그 캐릭터에 걸맞는 colorization을 진행하도록 하게 하는 layer입니다. 처음에 CNN을 통해 encoding된 vector에 classifier를 통하여 어떤 캐릭터인지 구분해주는 정보를 포함한 vector를 concatenate하는 형태로 구성되어 있습니다. Fusion layer에 더욱 관심이 있으신 분들은 위 링크를 참조하시면 되겠습니다. 

![alt text](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/images/model.png)

Note: Emil은 fusion layer를 도출하기 위한 classification algorithm으로 InceptionV3를 사용하였지만, 저는 ResNet을 사용해보았습니다.

### DCGAN
<p>(1) http://cs231n.stanford.edu/reports/2017/pdfs/302.pdf</p>
<p>(2) https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py</p>

2014년에 Ian Goodfellow에 의해 처음 소개된 이후로, GAN은 다양한 분야에서 많이 사랑받고 있는 딥러닝 알고리즘입니다. 최근에 읽은 (1) 논문에서는 이 GAN 알고리즘을 이용한 colorization algorithm을 구현했었습니다. 때문에, 저도 한번 도전해보고자 하는 마음으로 CNN 구조가 포함된 DCGAN을 이용하여 colorization algorithm을 구현해보고자 하였습니다. 가 작성한 Keras GAN 코드는 Erik Lindernoren의 Github을 참조하였습니다 (2).

![alt text](https://gluon.mxnet.io/_images/dcgan.png)
(이미지 출처: https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html)

### WGAN
<p>(1) https://arxiv.org/abs/1701.07875</p>
<p>(2) https://medium.com/@sunnerli/the-story-about-wgan-784be5acd84c</p>
<p>(3) https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i</p>
<p>(4) https://vincentherrmann.github.io/blog/wasserstein/(/p>

GAN이 유명해지면서 GAN자체에 집중하는 사람들이 생김과 동시에 GAN의 단점 자체에도 집중하는 사람들이 생기기 시작했습니다. 때문에, original GAN의 단점들을 보완해주는 새로운 버전의 GAN들이 등장하였고, Arjovsky와 Bottou (1)가 2017년 발표한 WGAN (Wasserstein GAN)은 이 중의 하나입니다. 일반 GAN에서 loss 값을 계산할 때 사용되는 KL과 JS Divergence를 사용하지 않고 새로운 metric인 Wasserstein Distance 개념을 적용한 GAN으로써, 기존의 KL, JS Divergence로는 계산하는 것이 불가능했던 loss 값들을 Wasserstein Distance metric으로 이용해 계산하여 GAN을 보완한 알고리즘이라고 할 수 있겠습니다. 

![alt text](https://cdn-images-1.medium.com/max/800/1*xRjphX2OGhfDllYFIkabzw.png "Brief Explanation of Divergence Metrics")

Wasserstein distance와 관련해서 궁금하신 분들은 위에 언급한 (2), (3)번 웹사이트를 참고하시면 좋을 것 같습니다. WGAN의 Keras구현을 위해서 저는 Vincent Herrmann의 code (4)를 참고하였습니다. WGAN의 이론적인 부분에 관해서는 추후에 따로 repo를 만들어서 제가 할 수 있는 한 최대한 설명해보려고 합니다.


## Acknowledgement
여러 질문들 및 막힐때마다 대답해주시고 최신 기계들을 지원해주신(무려 RTX 2080ti!!) Neowiz Play Studio와 오성규님(hanmaum@neowiz.com)께 특별히 감사드립니다. 

## Contact Information
<p>facebook: https://www.facebook.com/dabin.moon.7 </p>
<p>email: dabsdamoon@neowiz.com</p>
