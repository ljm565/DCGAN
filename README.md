# Deep Convolutional Generative Adversarial Network (DCGAN)
## 설명
Deep Convolutional Generative Adversarial Network (DCGAN)는 생성 모델인 vanilla GAN의 문제점을 극복하고, CNN이 black-box라는 것에 실험으로써 반박한 알고리즘입니다.
여기서는 DCGAN의 코드를 확인할 수 있으며, CelebA 데이터의 경우 DCGAN이 학습하면서 생성하는 순차적인 결과의 변화를 gif 형식의 파일로 가시화할 수 있습니다. 또한 학습된 모델이 생성한 데이터의 질을 확인하기 위하여 [Fréchet Inception Distance (FID) score](https://github.com/mseitzer/pytorch-fid)를 계산할 수 있습니다(출처: https://github.com/mseitzer/pytorch-fid). DCGAN에 대한 자세한 설명은 [Generative Adversarial Network (GAN)](https://ljm565.github.io/contents/DCGAN1.html)를 참고하시기 바랍니다.
<br><br><br>

## 모델 종류
* ### DCGAN
    Convolutional layer를 사용한 DCGAN 구현되어 있습니다.
<br><br><br>

## 사용 데이터
* 실험으로 사용하는 데이터는 [Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang Multimedia Laboratory의 CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 데이터입니다. CelebA는 유명인들의 얼굴을 모아놓은 데이터입니다.
* 학습 데이터의 경로를 설정하여 사용자가 가지고 있는 데이터도 학습 가능합니다.
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 FID score를 보고싶은 경우에는 test로 설정해야합니다. test를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m test 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test를 할 경우에도 test 할 모델의 이름을 입력해주어야합니다(최초 학습시 config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 학습 된 모델이 생성한 데이터의 FID score 계산 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 main.py -d cpu -m test -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    * **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**

    * ### CelebA 데이터 사용 시
        * CelebA_train: {0, 1} 중 선택, 0인 경우 사용자 지정 데이터, 1인 경우 CelebA 데이터 학습.
        <br><br>
    
    * ### 사용자 지정 데이터 사용 시
        * 사용자 지정 데이터 경로가 하나일 때(train/val/test set 구분이 없을 때)
            * custom_data_proportion: train/val/test 데이터 split 비율 설정(e.g. [0.7, 0.2, 0.1]), **비율 합이 1이 되어야 함.**
            * two_folders: 상관 없음.
            * train_data_path: 학습에 사용할 데이터가 있는 경로.
            * val_data_path: 상관 없음.
            * test_data_path: 상관 없음.
            <br><br>

        * 사용자 지정 데이터가 2개로 나눠져 있을 때(e.g train/test set) val set 생성
            * custom_data_proportion: 하나의 경로에 있는 데이터를 나눌 비율 설정(e.g. [0.7, 0.3]), **비율 합이 1이 되어야 함.**
            * two_folders: ['train', 'val'], ['val', 'test'] 둘 중 하나로 설정.
                * ['train', 'val']: train_data_path 에 있는 데이터를 custom_data_proportion에 맞춰서 train/val set으로 split.
                * ['val', 'test']: test_data_path 에 있는 데이터를 custom_data_proportion에 맞춰서 val/test set으로 split.
            * train_data_path: 학습에 사용할 데이터가 있는 경로.
            * val_data_path: 상관 없음.
            * test_data_path: 모델 결과를 테스트할 때 사용할 데이터가 있는 경로.
            <br><br>

        * 사용자 지정 데이터가 train/val/test set으로 모두 나눠져있을 때
            * custom_data_proportion: [1]로 설정.
            * two_folders: 상관 없음.
            * train_data_path: 학습에 사용할 데이터가 있는 경로.
            * val_data_path: validatin 할 때 사용할 데이터가 있는 경로.
            * test_data_path: 모델 결과를 테스트할 때 사용할 데이터가 있는 경로.
            <br><br>


    * base_path: 학습 관련 파일이 저장될 위치
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/model/{model_name}/{model_name}.pt 로 저장.
    * data_name: 사용자 지정 데이터 학습시 사용. 학습 데이터 이름 설정. base_path/{data_name}/ 내부에 train.pkl, val.pkl, test.pkl 파일로 저장. 전처리 등 시간 절약을 위해 이후 같은 data_name을 사용할 시 저장된 데이터를 불러서 사용(사용자 지정 데이터 사용시 data 폴더 생성). 
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * color_channel: 학습에 사용되는 데이터가 흑백이면 1, 칼라면 3으로 설정(CelebA 사용 시 3으로 설정).
    * img_size (8 이상의 수): 데이터의 전 처리 할 크기를 지정. 지정한 크기의 정사각 이미지로 학습 실행. **2의 제곱수로 지정 (e.g. 8, 16, 64, 128...)**
    * convert2grayscale: {0, 1} 중 선택, color_channel = 3 일때만 작동. 칼라 데이터를 흑백 데이터로 변경하고싶을 때 1, 칼라로 유지하고싶을 때 0으로 설정.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 지정.
    * lr: learning rate 지정.
    * beta1: Adam optimizer의 beta1 hyperparameter
    * g_dim: DCGAN의 generator 차원 지정.
    * d_dim: DCGAN의 discriminator 차원 지정.
    * noise_init_size: GAN의 generator가 학습할 때 랜덤으로 생성하는 데이터의 hidden dimension.
    * training_visualization: {0, 1} 중 선택, 1인 경우 GAN이 학습 하면서 생성한 데이터의 변화 과정이 담긴 이미지 및 gif 파일 저장, 0이면 저장 안함.
    * generation_img_folder_name: training_visualization이 1인 경우 학습 epoch별(10 epoch별) 생성한 이미지 저장할 폴더명, base_path/result/{generation_img_folder_name} 폴더에 이미지 저장됨.
    * generation_gif_name: training_visualization이 1인 경우 모든 epoch별 GAN이 생성한 이미지를 gif 형식으로 저장하는 데 사용될 파일명, base_path/result/{generation_gif_name}에 저장됨.
    <br><br>

    * ### 학습된 GAN 모델의 성능 평가 (FID score)
        * score_cal_folder_name: 학습된 모델을 평가하기 위해 FID score를 계산할 때, GAN이 생성한 데이터를 저장할 폴더명, base_path/test/{score_cal_folder}/fake에 GAN이 생성한 데이터가 저장됨.
        <br><br><br>


## 결과
* DCGAN 결과<br><br>
<img src="images/generation_gif_100epochs.gif" width="50%"><br><br>
<img src="images/RealandFake.png" width="100%"><br><br>
<br><br><br>


## License
© 2022. Jun-Min Lee. All rights reserved.<br>
ljm56897@gmail.com, ljm565@kaist.ac.kr, ljm565@naver.com

