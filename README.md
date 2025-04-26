# VQGAN-Transformer


# 1. First Create VQGAN

Create the VQGAN with vqgan_lt.py

# 2. Train the model GPT perso (transformer)

Run the file vqgan_gen_car.py , and launch only the function ```transtrainP(model, optimizer, device, scheduler, image_test, image_pos,  200)```.

# 3. Inference

Run the file vqgan_gen_car.py , and launch only the function ```image_generation_labelP(model, image_test, image_pos)```.

The other training function don't run well, it's some shit when I didn' t master the subject.

# 4. Dataset

Download a dataset from Kaggle ans create some folder dataset/class1, dataset/class2 , etc...
