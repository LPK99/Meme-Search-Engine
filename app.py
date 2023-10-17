import streamlit as st
import os
import clip
import torch
from PIL import Image, ImageFile
import faiss
import json
import numpy as np

HOME_DIR = "images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
images = []
ImageFile.LOAD_TRUNCATED_IMAGES = True

@st.cache_resource
def retrieve_embeddings():
  if os.path.exists("index.bin"):
      index = faiss.read_index("index.bin")
      with open("references.json", "r") as f:
          data = json.load(f)
  else:
      index = faiss.IndexFlatL2(512)

      images = []

      for item in os.listdir(HOME_DIR):
          if item.lower().endswith((".jpg", ".jpeg", ".png")):
              image = (
                  preprocess(Image.open(os.path.join(HOME_DIR, item)))
                  .unsqueeze(0)
                  .to(device)
              )
              images.append((item, image))
          else:
              continue

      data = []

      for i in images:
          with torch.no_grad():
              image_features = model.encode_image(i[1])
              image_features /= image_features.norm(dim=-1, keepdim=True)

              data.append(
                  {
                      "image": i[0],
                      "features": np.array(image_features.cpu().numpy()).tolist(),
                  }
              )

              index.add(image_features.cpu().numpy())

      faiss.write_index(index, "index.bin")

      with open("references.json", "w") as f:
          json.dump(data, f)

  return index, data

def main():
    st.title('Meme Search Engine')
    query = st.text_input('Enter your query')
    tokenized_query = clip.tokenize([query]).to(device)
    if st.button('Search'):
        with torch.no_grad():
            text_features = model.encode_text(tokenized_query)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            index, data = retrieve_embeddings()

            D, I = index.search(text_features.cpu().numpy(), k=3)
    
            for i in I[0]:
                image = Image.open(os.path.join(HOME_DIR, data[i]["image"]))
                st.image(image=image)
                st.caption(data[i]['image'])
                
if __name__ == '__main__':
    main()