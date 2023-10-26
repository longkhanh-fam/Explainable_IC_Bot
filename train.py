import tensorflow as tf
import keras
from model import *
import pickle

def masked_loss(labels, preds):  
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

  mask = (labels != 0) & (loss < 1e8) 
  mask = tf.cast(mask, loss.dtype)

  loss = loss*mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_acc(labels, preds):
  mask = tf.cast(labels!=0, tf.float32)
  preds = tf.argmax(preds, axis=-1)
  labels = tf.cast(labels, tf.int64)
  match = tf.cast(preds == labels, mask.dtype)
  acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
  return acc
class GenerateText(tf.keras.callbacks.Callback):
  def __init__(self):
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
    self.image = load_image(image_path)

  def on_epoch_end(self, epochs=None, logs=None):
    print()
    print()
    print()

@Captioner.add_method
def simple_gen(self, image, temperature=1):
  initial = self.word_to_index([['[START]']]) # (batch, sequence)
  img_features = self.feature_extractor(image[tf.newaxis, ...])

  tokens = initial # (batch, sequence)
  for n in range(50):
    preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
    preds = preds[:,-1, :]  #(batch, vocab)
    if temperature==0:
        next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
    else:
        next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
    tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

    if next[0] == self.word_to_index('[END]'):
      break
  words = index_to_word(tokens[0, 1:-1])
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  return result.numpy().decode()

if __name__ == "__main__":
    train_raw, test_raw = flickr8k()

    train_ds = load_dataset('train_cache')
    #test_ds = load_dataset('test_cache')


    #load
    from_disk = pickle.load(open("tv_layer.pkl", "rb"))
    new_v = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])


    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        ragged=True)
    tokenizer.adapt(train_raw.map(lambda fp,txt: txt).unbatch().batch(1024))

    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)
    
    #save tokenizer
    # print(tokenizer("this"))
    # pickle.dump({'config': tokenizer.get_config(),
    #             'weights': tokenizer.get_weights()}
    #             , open("tv_layer.pkl", "wb"))


    # Create mappings for words to indices and indices to words.



    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
    # This might run a little faster if the dataset didn't also have to load the image data.
    output_layer.adapt(train_ds.map(lambda inputs, labels: labels))

    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
    
    
    






    


    # g = GenerateText()
    # g.model = model
    # g.on_epoch_end(0)

    # callbacks = [
    #     GenerateText(),
    #     tf.keras.callbacks.EarlyStopping(
    #         patience=5, restore_best_weights=True)]

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #         loss=masked_loss,
    #         metrics=[masked_acc])

    # history = model.fit(
    #     train_ds.repeat(),
    #     steps_per_epoch=50,
    #     validation_data=test_ds.repeat(),
    #     validation_steps=20,
    #     epochs=100,
    #     callbacks=callbacks)



    model.load_weights('./checkpoints1/my_checkpoint')

    image_path = './anh-em-quoc-co-quoc-nghiep-2-413.png'
    #image_path = tf.keras.utils.get_file(origin=image_url)
    image = load_image(image_path)
    result = model.simple_gen(image, temperature=0.0)
    print(result)
    #model.save('my_model.keras')
    #model.save_weights('./checkpoints/my_checkpoint')
