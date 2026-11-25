import asyncio
from mcp.server.fastmcp import FastMCP
from functools import lru_cache
from pathlib import Path
from io import BytesIO
from threading import Lock
from multimodal import index, ROOT
from common import model
from PIL import Image
import base64
mcp=FastMCP('sav',host='127.0.0.1',port=23000)
index_lock = Lock()
@lru_cache(maxsize=1)
def resources(): return index()


def image_content(path: Path) -> dict:
    buffer = BytesIO()
    with Image.open(path) as image:
        image.thumbnail((1600, 1600))
        image.convert('RGB').save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}}


@mcp.tool()
async def rag_multimodal(question: str, image_ref: str = '') -> str:
    """Recherche texte et figures dans les manuels; image_ref est une référence locale autorisée."""
    return await asyncio.to_thread(search_manuals, question, image_ref)


def search_manuals(question: str, image_ref: str) -> str:
    photo=None
    if image_ref:
        if not image_ref.replace('_','').isalnum(): return 'Référence invalide.'
        photo=ROOT/'data'/f'{image_ref}.png'
        if not photo.exists(): return 'Photo introuvable.'
    with index_lock:
        col,embed=resources()
    if not col.count(): return 'Aucun manuel indexé.'
    figures: dict[tuple[str, int, str], Path] = {}
    passages: dict[tuple[str, int, str], None] = {}

    vectors = [embed(text=question)]
    if photo:
        with Image.open(photo) as image:
            vectors.append(embed(image=image))
    for kind, limit in [('text', 5), ('image', 2)]:
        result = col.query(query_embeddings=vectors, n_results=limit, where={'type': kind})
        for docs, metas in zip(result['documents'], result['metadatas']):
            for doc, meta in zip(docs, metas):
                source, page = meta['source'], meta['page']
                if kind == 'text':
                    passages[source, page, doc] = None
                else:
                    path = Path(meta['path'])
                    if path.is_file():
                        figures[source, page, str(path)] = path
    if not passages and not figures:
        return 'Aucun passage ni figure trouvé dans les manuels.'

    evidence = '\n'.join(f'{source}, page {page} — {text}' for source, page, text in passages)
    content = [{'type': 'text', 'text': f'Question\n{question}\n\nPassages des manuels\n{evidence}'}]
    if photo:
        content.extend([{'type': 'text', 'text': 'Photo jointe par le client'}, image_content(photo)])
    for (source, page, _), path in list(figures.items())[:2]:
        content.extend([{'type': 'text', 'text': f'Figure du manuel {source}, page {page}'}, image_content(path)])
    answer = model('vision' if photo or figures else 'default').invoke([
        ('system', 'Réponds uniquement à partir des passages et des figures fournis. Cite le manuel et la page. '
         'Si les éléments sont insuffisants, indique-le. Si la photo ne montre aucun appareil ou symptôme identifiable, '
         'demande le modèle de l’appareil et une photo nette de son écran ou de son voyant. '
         'Les données de démonstration sont fictives.'),
        ('user', content),
    ])
    return str(answer.content)
if __name__=='__main__': mcp.run(transport='streamable-http')
