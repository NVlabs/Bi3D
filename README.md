## Bi3D &mdash; Official PyTorch Implementation

![Teaser image](imgs/teaser.png)

**Bi3D: Stereo Depth Estimation via Binary Classifications**<br>
Abhishek Badki, Alejandro Troccoli, Kihwan Kim, Jan Kautz, Pradeep Sen, Orazio Gallo<br>
IEEE CVPR 2020<br>

## Abstract: 
*Stereo-based depth estimation is a cornerstone of computer vision, with state-of-the-art methods delivering accurate results in real time. For several applications such as autonomous navigation, however, it may be useful to trade accuracy for lower latency. We present Bi3D, a method that estimates depth via a series of binary classifications. Rather than testing if objects are* at *a particular depth D, as existing stereo methods do, it classifies them as being* closer *or* farther *than D. This property offers a powerful mechanism to balance accuracy and latency. Given a strict time budget, Bi3D can detect objects closer than a given distance in as little as a few milliseconds, or estimate depth with arbitrarily coarse quantization, with complexity linear with the number of quantization levels. Bi3D can also use the allotted quantization levels to get continuous depth, but in a specific depth range. For standard stereo (i.e., continuous depth on the whole range), our method is close to or on par with state-of-the-art, finely tuned stereo methods.*


## Paper:
https://arxiv.org/pdf/2005.07274.pdf<br>

## Videos:<br>
<a href="https://www.youtube.com/watch?v=HuEwjpw5O64&feature=youtu.be">
  <img src="https://img.youtube.com/vi/HuEwjpw5O64/0.jpg" width="300"/>
</a>
<a href="https://www.youtube.com/watch?v=UfvUny4pdMA&feature=youtu.be">
  <img src="https://img.youtube.com/vi/UfvUny4pdMA/0.jpg" width="300"/>
</a>
<a href="https://www.youtube.com/watch?v=Ifgcm6VI3NE&feature=youtu.be">
  <img src="https://img.youtube.com/vi/Ifgcm6VI3NE/0.jpg" width="300"/>
</a>

## Citing Bi3D:
    @InProceedings{badki2020Bi3D,
    author = {Badki, Abhishek and Troccoli, Alejandro and Kim, Kihwan and Kautz, Jan and Sen, Pradeep and Gallo, Orazio},
    title = {Bi3D: Stereo Depth Estimation via Binary Classifications},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020}
    }

## Code:<br>
Code coming soon. 
