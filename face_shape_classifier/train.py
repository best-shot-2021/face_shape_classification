from trainer import FaceShapeTrainer

face_trainer = FaceShapeTrainer(model_name='classification_model_v3.h5', batch_size=64, num_of_epoch=40)
face_trainer.train()
