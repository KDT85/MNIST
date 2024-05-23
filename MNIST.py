import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pygame

def load_and_preprocess_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

def create_and_train_model(train_images, train_labels, test_images, test_labels):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save('mnist_cnn_model.h5')
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('MNIST CNN Model Accuracy')
    plt.show()
    
    print("Model training complete and saved as 'mnist_cnn_model.h5'")

def predict_digit(model, image):
    image = pygame.surfarray.array3d(image)
    image = np.mean(image, axis=2)  # Convert to grayscale
    image = image.T  # Transpose to match the expected input shape
    image = pygame.transform.scale(pygame.surfarray.make_surface(image), (28, 28))  # Resize to 28x28
    image = pygame.surfarray.array3d(image)
    image = np.mean(image, axis=2)  # Convert to grayscale again if needed
    image = image.reshape((1, 28, 28, 1)).astype('float32') / 255  # Reshape and normalize
    prediction = model.predict(image)
    return np.argmax(prediction)

def run_pygame_app():
    pygame.init()
    window_size = 280
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Draw a Digit")

    drawing = False
    last_pos = None
    color = (255, 255, 255)
    radius = 10

    model = load_model('mnist_cnn_model.h5')

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos
            
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                prediction = predict_digit(model, window)
                print("Predicted Digit:", prediction)
            
            if event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(window, color, event.pos, radius)
                if last_pos is not None:
                    pygame.draw.line(window, color, last_pos, event.pos, radius * 2)
                last_pos = event.pos

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    window.fill((0, 0, 0))
        
        pygame.display.flip()

    pygame.quit()

def main():
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_mnist()
    create_and_train_model(train_images, train_labels, test_images, test_labels)
    run_pygame_app()

if __name__ == "__main__":
    main()

