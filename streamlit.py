import streamlit as st
from PIL import Image
from model import *
import tempfile
import os
import pickle

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
    st.title("Web APp")

    #train_ds = load_dataset('train_cache')

    from_disk = pickle.load(open("./tv_layer.pkl", "rb"))
    tokenizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    tokenizer.set_weights(from_disk['weights'])

    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)

    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
    # This might run a little faster if the dataset didn't also have to load the image data.
    #output_layer.adapt(train_ds.map(lambda inputs, labels: labels))


    output_layer.load_weights('token_output_weights.npy')

    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)

    model.load_weights('./checkpoints1/my_checkpoint')



    img_upload = st.file_uploader(label='Upload Image', type=['png', 'jpg'])
    


    #img_pt = "imgs/banmuonhenho-783.jpg" #if img_upload is None else img_upload










    if img_upload is not None:
        # Create a temporary directory to save the uploaded image
        temp_dir = tempfile.TemporaryDirectory()
        temp_image_path = os.path.join(temp_dir.name, img_upload.name)

        # Save the uploaded image to the temporary directory
        with open(temp_image_path, 'wb') as f:
            f.write(img_upload.read())

        # Load and process the image
        image = load_image(temp_image_path)


        # Clean up the temporary directory
        temp_dir.cleanup()


    if st.button('Generate captions!'):
        #image = load_image(img_upload)
        result = model.simple_gen(image, temperature=0.0)
        st.write(result)
