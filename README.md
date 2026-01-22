### APRI IN EDIT COSì DA VEDERE INDENTAZIONE CORRETTA

# Link Dowload per Dataset Immagini:

andrewmvd: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?resource=download \
trngvhong: https://www.kaggle.com/datasets/trngvhong/mfvt-dataset?utm_source=chatgpt.com \
yolo-6hrdo: https://universe.roboflow.com/yolo-6hrdo/facemask-xafme

# Struttura Folder Adottata: 

```
mask-yolo-2stage/
  data/
    raw/
      andrewmvd_voc/
        annotation/
        images/
      mfvt_coco/
        test/
        images/
        class.txt
      roboflow_yolo/
        test/
        train/
        valid/
    processed/
      base_yolo/
      person_crops_yolo/
  datasets/
    base_yolo.yaml
    person_crops_yolo.yaml
  src/
    prepare_base_yolo.py
    make_person_crops.py
    train_mask_yolo.py
    infer_pipeline.py
    shell_pipe_orchestrator.py
```
