# HuBMAP + HPA - Hacking the Human Body

🟧 [About Competition](https://www.kaggle.com/competitions/hubmap-organ-segmentation) <220623~220916>


[![colab-badge](https://user-images.githubusercontent.com/79159191/181145349-7c08a358-bcc2-4b9f-85db-04cb77e8fb84.svg)](https://colab.research.google.com/drive/1403u-rk3O0Xy661AzD6xGG5OZfGDGe78#scrollTo=_MnjUHohli5H)


-> My score 0.68, 175th 220826

-> My score 0.73, 175th 220830

-> My score 0.74, 210th 220903

-> My score 0.79 89th 220921 🥉

-> My score 0.80 97th 220922 🥉

🔥 Private Score

<img width="1045" alt="스크린샷 2022-09-23 오전 9 17 03" src="https://user-images.githubusercontent.com/79159191/191872451-f1b2c1d6-f20f-4ace-b992-f12c2aa758e7.png">



My goal is a 🥉. If it goes well, 🥈

trying pixel_size & np.float32/255 (segformer_b5)... 220829

trying stain h&e & out of mmseg & to scratch(segformer_b5)... 220830

i tried some augmentation about color because hpa & hubmap is diffent about color 
it is meaningful... 220830

i apply pixel_size augmentation but it be ineffective... 220903

about tiling

[Systematic Evaluation of Image Tiling Adverse Effects on Deep Learning Semantic Segmentation](https://www.frontiersin.org/articles/10.3389/fnins.2020.00065/full)

i trying batch_size, loss fuction, augmentation... 220909

changed decoder-head to mixupsample... 220910

studying swa... 220913

train with coat... 220919

i train almost 48 hour two model

0.4 normalise and 0.7 normalise 

i got hubmap 0.58 hpa 0.21

My score 0.79++

i'm looking for hpa 22 or 23 model ... 220921

last day i train another model normalise 0.8~

and finding best thershold... 220922

i got 0.80 with three ensemble... 220922


