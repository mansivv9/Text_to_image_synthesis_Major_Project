# Text_to_image_synthesis_Major_Project
1. **Data Processing**

-> BIRDS (Create a directory data/birds)

* Download images from [here](https://data.caltech.edu/records/20098) and place them in data/birds.
* Download preprocessed metadata from [here](https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ/view) and save it in data/birds.

-> FLOWERS (Create a directory data/flowers)

* Download images from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) 
* Download the preprocessed captions.pickle file from [here](https://drive.google.com/file/d/1T7xvD2qEU4PkrQqFTxh6ZZyEYGFjmBIu/view?usp=share_link) and save it to data/flowers
* Download other annotations and dictionary files required for training from [here](https://drive.google.com/drive/folders/1Y9rOLl3-FwQp0Y9sYbHM_D5JVrnPjD5h?usp=share_link) and save them in data/flowers

2. **Clone the repository**
    
        git clone https://github.com/mansivv9/Text_to_image_synthesis_Major_Project.git
        cd 'Text_to_image_synthesis_Major_Project/AttnGAN+CL+SN+RoBERTa(AttnGAN_V2)'

3. **DAMSM Training phase**
* First give the complete path of data/birds or data/flowers in the field "DATA_DIR" in cfg/DAMSM/bird.yml or flowers.yml files.
* Birds       

        python code/pretrain_DAMSM.py --cfg code/cfg/DAMSM/bird.yml --gpu 0
* Flowers

        python code/pretrain_DAMSM.py --cfg code/cfg/DAMSM/flowers.yml --gpu 0

4. **GAN Training phase**
* First give the complete path of data/birds or data/flowers in the field "DATA_DIR" in cfg/bird_attn2.yml or flowers_attn2.yml files.
* Birds

        python code/main.py --cfg code/cfg/bird_attn2.yml --gpu 0
* Flowers

        python code/main.py --cfg code/cfg/flowers_attn2.yml --gpu 0
5. **Evaluation**

-> First give the complete path of data/birds or data/flowers in the field "DATA_DIR" in code/cfg/eval_bird.yml or code/cfg/eval_flowers.yml.

-> Give the path of best model checkpoint (eg: netG_epoch_200.pth) in "NET_G" field of code/cfg/eval_bird.yml or eval_flowers.yml.

-> give the path of pretrained text encoder (eg: text_encoder200.pth) in "NET_E" field of code/cfg/eval_bird.yml or eval_flowers.yml.

-> In this experiment, 30000 images are generated which are conditioned on the captions in the test set. These are considered as fake images and are used for the calculation of Inception score, FID and R-Precision.

-> Generation of images in the validation set and **R-precision calculation**
* Birds

  -> set B_VALIDATION: True in code/cfg/eval_bird.yml

        python code/main.py --cfg code/cfg/eval_bird.yml --gpu 0
* Flowers

  -> set B_VALIDATION: True in code/cfg/eval_flowers.yml

        python code/main.py --cfg code/cfg/eval_flowers.yml --gpu 0


-> **Inception score**
  
* give the path of "inception_finetuned_models/birds_valid299/model.ckpt" in inception_score_birds.py or "inception_finetuned_models/flowers_valid299/model.ckpt"in inception_score_flowers.py file in checkpoint_dir.
* give the path of generated images on the validation set "val_gen_images/valid/single" in image_folder in inception_score_birds.py or inception_score_flowers.py file.

* Birds

        python eval/IS/inception_score_birds.py
* Flowers

        python eval/IS/inception_score_flowers.py

-> **FID score**

*  The entire datasets consisting of 11788 and 8189 images were considered as real images for the calculation of FID over the CUB and Flowers dataset respectively.
* Birds
        
        python eval/FID/fid_score.py --gpu 0 --batch-size 50 --path1 'data/birds/CUB_200_2011/images' --path2 'val_gen_images/valid/single'

* Flowers

        python eval/FID/fid_score.py --gpu 0 --batch-size 50 --path1 'data/flowers/jpg' --path2 'val_gen_images/valid/single'


-> Generation of images from example captions

* First set B_VALIDATION: False in code/cfg/eval_bird.yml or code/cfg/eval_flowers.yml.
* Here example_captions are used to generate images using our model.
* Birds

        python code/main.py --cfg code/cfg/eval_bird.yml --gpu 0
* Flowers

        python code/main.py --cfg code/cfg/eval_flowers.yml --gpu 0
        
6. Experiments

We have included the changes to be made to perform the experiments mentioned in this research work in [code](https://github.com/mansivv9/Text_to_image_synthesis_Major_Project/blob/main/attngan-experiments.ipynb)

* Experiment 1: We tried transformers like AlBERT
and RoBERTa as text encoders instead of BiLSTM in
the AttnGAN architecture. Contrastive learning was
applied in the GAN training phase. 

* Experiment 2: In order to stabilize the training
of GANs, we applied spectral normalization to the
discriminator of AttnGAN architecture. Following the
success of the previous Experiment 1, we adopted
RoBERTa transformer as text encoder and contrastive
learning technique in the GAN training phase. 

* Experiment 3: An augmented model which uses RoBERTa
as text encoder, contrastive learning in both DAMSM
and GAN training phase and spectral normalization
in the discriminator. Just clone this repository and follow the steps mentioned above to perform this experiment.


7. **Data and Pretrained models**

* The additional data required for flowers, inceptionV3 fine tuned models for the calculation of inception score, pretrained DAMSM models and that of GAN training phase are provided here: [click here](https://drive.google.com/drive/folders/1P7CYyYIDv5lEcVhDYAdf0hRMu9zLtU81?usp=share_link)

8. **REFERENCES** All the references used in this research work have been cited here

[1] Kashyap Kathrani. All about embeddings. Apr. 2022.
URL: https://medium.com/@kashyapkathrani/
all-about-embeddings-829c8ff0bf5b.

[2] Ashish Vaswani et al. “Attention is all you need”. In:
Advances in neural information processing systems
30 (2017).

[3] Tao Xu et al. “Attngan: Fine-grained text to image
generation with attentional generative adversarial
networks”. In: Proceedings of the IEEE conference
on computer vision and pattern recognition. 2018,
pp. 1316–1324.

[4] Takeru Miyato et al. “Spectral normalization for
generative adversarial networks”. In: arXiv preprint
arXiv:1802.05957 (2018).

[5] Tan M Dinh, Rang Nguyen, and Binh-Son Hua.
“TISE: Bag of Metrics for Text-to-Image Synthesis
Evaluation”. In: Computer Vision–ECCV 2022: 17th
European Conference, Tel Aviv, Israel, October 23–
27, 2022, Proceedings, Part XXXVI. Springer. 2022,
pp. 594–609.

[6] Hui Ye et al. “Improving text-to-image synthesis using contrastive learning”. In: arXiv preprint
arXiv:2107.02423 (2021).

[7] Yinhan Liu et al. “Roberta: A robustly optimized bert pretraining approach”. In: arXiv preprint
arXiv:1907.11692 (2019).

[8] Tim Salimans et al. “Improved techniques for training gans”. In: Advances in neural information processing systems 29 (2016).

[9] Jason Brownlee. How to implement the Frechet Inception Distance (FID) for evaluating Gans. Oct.
2019. URL: https : / / machinelearningmastery .
com / how - to - implement - the - frechet -
inception-distance-fid-from-scratch/.

[10] Min Jin Chong and David Forsyth. “Effectively unbiased fid and inception score and where to find
them”. In: Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. 2020,
pp. 6070–6079.

[11] Catherine Wah et al. “The caltech-ucsd birds-200-
2011 dataset”. In: (2011).

[12] Maria-Elena Nilsback and Andrew Zisserman. “Automated flower classification over a large number
of classes”. In: 2008 Sixth Indian Conference on
Computer Vision, Graphics & Image Processing. IEEE.
2008, pp. 722–729.

[13] Pouget-Abadie Goodfellow, Xu Mirza, Ozair WardeFarley, et al. “Goodfellow I”. In: Pouget-Abadie J.,
Mirza M., Xu B., Warde-Farley D., Ozair S., Courville
A., Bengio Y., Generative adversarial nets, Advances
in neural information processing systems 27 (2014).

[14] Scott Reed et al. “Generative adversarial text to
image synthesis”. In: International conference on
machine learning. PMLR. 2016, pp. 1060–1069.

[15] Han Zhang et al. “Stackgan: Text to photo-realistic
image synthesis with stacked generative adversarial
networks”. In: Proceedings of the IEEE international
conference on computer vision. 2017, pp. 5907–5915.

[16] Han Zhang et al. “Stackgan++: Realistic image synthesis with stacked generative adversarial networks”.
In: IEEE transactions on pattern analysis and machine intelligence 41.8 (2018), pp. 1947–1962.

[17] M Siddharth and R Aarthi. “Text to image gans with
roberta and fine-grained attention networks”. In:
International Journal of Advanced Computer Science
and Applications 12.12 (2021).

[18] Minfeng Zhu et al. “Dm-gan: Dynamic memory
generative adversarial networks for text-to-image
synthesis”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
2019, pp. 5802–5810.

[19] Mike Schuster and Kuldip K Paliwal. “Bidirectional
recurrent neural networks”. In: IEEE transactions on
Signal Processing 45.11 (1997), pp. 2673–2681.

[20] Christian Szegedy et al. “Rethinking the inception
architecture for computer vision”. In: Proceedings of
the IEEE conference on computer vision and pattern
recognition. 2016, pp. 2818–2826.

[21]  https://github.com/taoxugit/AttnGAN.

[22] https://github.com/huiyegit/T2I_CL.

[23] https://github.com/VinAIResearch/tise-toolbox.

[24] https://github.com/MinfengZhu/DM-GAN

[25] https://github.com/hanzhanggit/StackGAN-inception-model

[26] https://github.com/senmaoy/RAT-GAN








