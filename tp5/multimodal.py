from pathlib import Path
import hashlib
ROOT=Path(__file__).parent

def index():
    import fitz, chromadb, open_clip, torch
    from PIL import Image
    encoder, _, prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    encoder.eval(); tokenizer=open_clip.get_tokenizer('ViT-B-32')
    def embed(text=None, image=None):
        with torch.no_grad():
            v=encoder.encode_text(tokenizer([text])) if image is None else encoder.encode_image(prep(image.convert('RGB')).unsqueeze(0))
            return torch.nn.functional.normalize(v,dim=-1)[0].tolist()
    db=chromadb.PersistentClient(path=str(ROOT/'.chroma'))
    col=db.get_or_create_collection('manuals',metadata={'hnsw:space':'cosine'})
    out=ROOT/'data/figures';out.mkdir(exist_ok=True)
    for path in (ROOT/'data').glob('*.pdf'):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with fitz.open(path) as doc:
            for i,page in enumerate(doc,1):
                words=page.get_text().split()
                for j in range(0,len(words),25):
                    text=' '.join(words[j:j+25]); col.upsert(ids=[f'{digest}-{i}-t{j}'], embeddings=[embed(text=text)],documents=[text],metadatas=[{'source':path.name,'page':i,'type':'text','path':''}])
                for j,img in enumerate(page.get_images(full=True)):
                    data=doc.extract_image(img[0]); target=out/f'{digest[:12]}-{i}-{j}.{data["ext"]}';target.write_bytes(data['image'])
                    with Image.open(target) as im: v=embed(image=im)
                    col.upsert(ids=[f'{digest}-{i}-i{j}'],embeddings=[v],documents=['Figure du manuel'],metadatas=[{'source':path.name,'page':i,'type':'image','path':str(target)}])
    return col,embed
