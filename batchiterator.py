
import torch
from utils import *
import numpy as np
#from evaluation import *
import tensorflow as tf
import keras







def BatchIterator(model, phase,
        Data_loader,optimizer , loss_fn):


    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 1000
    running_loss = 0.0
    total_loss = 0.0
    # print(Data_loader,"hello world line 25")
    
    for j, data in enumerate(Data_loader):


        imgs, labels, _ = data
        
        # print(imgs, labels, _ )

        batch_size = imgs.shape[0]
        batch_loss=0
        batch_accuracy=0
        
        imgs = imgs.permute(0, 2, 3,1)
        images_numpy = np.array(imgs)  # Assuming images is on GPU, move to CPU first
        labels_numpy = np.array(labels)
        

        if phase == "train":
            # optimizer.zero_grad()
            # model.train()
            # outputs = model(imgs)
            
            # n_members = 5
            # members = load_all_models(n_members)
            # print('Loaded %d models' % len(members))
            # stackedX = stacked_dataset(members, inputX)
            
            # resized_images = tf.image.resize(imgs, (224, 224))

            # Ensure the images have 3 channels
            # resized_images = tf.tile(resized_images, [1, 1, 1, 3])
            # print(resized_images,"resized images")
            
            # print(images_numpy,"images numpy")
            
            
            
           
            # Assuming model is your Keras functional model
            # Assuming batch_images is your batch of images with shape (batch_size, image_height, image_width, num_channels)
            # Assuming batch_labels is your batch of corresponding target labels

            # Initialize gradients accumulator
            gradients_accumulator = None
            

            # Iterate over each image in the batch
            for i in range(batch_size):
                # Extract the current image and corresponding label
                image = np.expand_dims(images_numpy[i], axis=0)  # Expand dimensions to make it a batch of size 1
                label = np.expand_dims(labels_numpy[i],axis=0) # Expand dimensions to make it a batch of size 1
                # print(label.shape)
                # print(label)
                # Compute gradients for the current image
                with tf.GradientTape() as tape:
                    predictions = model(image)
                    print(predictions,"predictions")
                    print(tf.convert_to_tensor(label),"label")
                    loss = loss_fn(tf.convert_to_tensor(label), predictions)
                    total_loss+=loss.numpy()

                gradients = tape.gradient(loss, model.trainable_variables)
                
                
                # if gradients_accumulator is None:
                #     gradients_accumulator = gradients
                # else:
                #     gradients_accumulator = [tf.add(gradient, grad_accum) for gradient, grad_accum in zip(gradients, gradients_accumulator)]

                # Accumulate gradients
                # if gradients_accumulator is None:
                #     gradients_accumulator = gradients
                # else:
                #     gradients_accumulator = [tf.add(gradient, grad_accum) for gradient, grad_accum in zip(gradients, gradients_accumulator)]

            # Update model parameters using accumulated gradients
            # print(gradients,"gradients")
            print(loss.numpy(),"loss")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(str(i * j))
            batch_loss = total_loss / batch_size       
            
                        
            # with tf.GradientTape() as tape:
            # # Run the forward pass of the layer.
            # # The operations that the layer applies
            # # to its inputs are going to be recorded
            # # on the GradientTape.
            #     logits = model(imgs, training=True)  # Logits for this minibatch

            # # Compute the loss value for this minibatch.
            #     loss_value = loss_fn(labels, logits)
            
            # grads = tape.gradient(loss_value, model.trainable_weights)

            # # Run one step of gradient descent by updating
            # # the value of the variables to minimize the loss.
            # optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # for i in range(batch_size):
            #     # Get the current image and label
            #     image = images_numpy[i:i+1]  # Take one image at a time
            #     label = labels_numpy[i:i+1]
                
            #     # Perform forward pass and compute loss
            #     loss, accuracy = model.train_on_batch(image, label)
                
            #     # Accumulate losses and accuracies
            #     total_loss += loss
            #     total_accuracy += accuracy
                
            # batch_loss = total_loss / batch_size
            # batch_accuracy = total_accuracy / batch_size 
            # batch_loss, batch_accuracy = model.train_on_batch(images_numpy, labels_numpy)
        
        else:
            for i in range(batch_size):
                # Extract the current image and corresponding label
                image = np.expand_dims(images_numpy[i], axis=0)  # Expand dimensions to make it a batch of size 1
                label = np.expand_dims(labels_numpy[i],axis=0) # Expand dimensions to make it a batch of size 1
                # print(label.shape)
                # print(label)
            
                output = model.predict(image)
                loss = loss_fn(tf.convert_to_tensor(label), output)
                total_loss+=loss
                
            batch_loss = total_loss / batch_size       
            

            # model.eval()
            # with torch.no_grad():
            #     outputs = model(imgs)


        # loss = criterion(outputs, labels)

        # if phase == 'train':

            # loss.backward()
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)
            # optimizer.step()  # update weights

        # running_loss += loss * batch_size
        # epoch_accuracy += batch_accuracy
        if (j % 64 == 0):
            print(str(j * batch_size))



    print(total_loss)
    return total_loss
