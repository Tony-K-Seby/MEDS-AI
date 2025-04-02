import time
import tensorflow as tf

start_time = time.time()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
end_time = time.time()

print(f"Keras modules loaded in {end_time - start_time:.2f} seconds")
