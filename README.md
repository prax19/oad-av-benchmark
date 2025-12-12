# OAD benchmark
A small framework created to compare some *SOTA* *online action detection* methods, with focus on appliance in *ADAS* and *ADS*.

## Setup instructions
Consider using `conda` or `Docker` to isolate your other projects or applications you use. Take a note, that Python version must be `3.11` for this setup.

1. Install the proper `PyTorch` wheel for your hardware:
    - [CUDA / CPU](https://pytorch.org/get-started/locally/)
    - [XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
2. ~~Run `pip install -r requirements.txt` to install basic requirements~~
3. Run `python setup/get_mmaction2.py` to install [`MMAction2`](https://github.com/open-mmlab/mmaction2)
4. *(optional)* Download `ROAD_Waymo` from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_road_plus_plus)
    - Download `videos` and `road_waymo_trainval_v1.0.json` with [gsutil](https://docs.cloud.google.com/storage/docs/gsutil_install), and place these files in `./data/road_waymo`
        ``` Bash
        mkdir "data/road_waymo"
        cd "data/road_waymo"
        gcloud auth login
        gsutil -m cp -r "gs://waymo_open_dataset_road_plus_plus/road_waymo_trainval_v1.0.json" "gs://waymo_open_dataset_road_plus_plus/videos" .
        cd ../..
        ```
        **You need to [register yourself](https://waymo.com/open/download/) and accept terms of use before downloading.**
5. Run `python setup/get_road.py`
    - `./data/` file structure should be following (including `ROAD_Waymo`):
        ```
        road-waymo/
            - road_trainval_v1.0.json
            - videos/
                - 2014-06-25-16-45-34_stereo_centre_02.mp4
                - 2014-06-26-09-53-12_stereo_centre_02.mp4
                - 2014-07-14-14-49-50_stereo_centre_01.mp4
                ...
        road-waymo/
            - road_waymo_trainval_v1.0.json
            - videos/
                - Train_00000.mp4
                - Train_00001.mp4
                - Train_00002.mp4
                ...
        ```

[comment]: <> (Continue with feature extraction)