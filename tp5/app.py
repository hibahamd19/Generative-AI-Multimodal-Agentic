import asyncio
import logging
from io import BytesIO
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from multimodal import ROOT
from sav_graph import graph as sav_graph

st.set_page_config(page_title='Assistance SAV', page_icon='🛠️')

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 75


async def answer_request(question, image_ref):
    return await asyncio.wait_for(
        sav_graph.ainvoke({'question': question, 'image_ref': image_ref}),
        timeout=REQUEST_TIMEOUT,
    )


st.html("""
<style>
    [data-testid="stMainBlockContainer"] {
        max-width: 56rem;
        padding: clamp(3.5rem, 8vw, 5rem) clamp(1rem, 4vw, 2.5rem) 2rem;
    }
    .sav-header {
        padding: clamp(1.25rem, 4vw, 2rem);
        border-radius: 1.25rem;
        background: linear-gradient(120deg, #102c28, #185043);
        color: #ffffff;
    }
    .sav-header h1 {
        padding: 0;
        margin: 0 0 .2rem;
        color: inherit;
        font-size: clamp(1.65rem, 5vw, 2.25rem);
        line-height: 1.15;
        letter-spacing: -.035em;
    }
    .sav-header p { margin: 0; color: #d1e8e0; }
    .sav-guide {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .85rem;
        margin: .4rem 0 0;
    }
    .sav-guide > div {
        background: #ffffff;
        color: #172d29;
        border: 1px solid #d6e2de;
        border-radius: 1rem;
        padding: 1rem 1.15rem;
    }
    .sav-guide p { margin: .3rem 0 0; color: #435b55; font-size: .9rem; }
    [data-testid="stChatMessage"] {
        border: 1px solid #d6e2de;
        border-radius: 1rem;
        padding: 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #e5f2ed;
        color: #172d29;
        border-color: #c9e1d7;
    }
    [data-testid="stChatMessage"] img {
        border-radius: .75rem;
        max-width: 100%;
    }
    [data-testid="stChatInput"] { max-width: 51rem; margin-inline: auto; }
    @media (max-width: 520px) {
        .sav-guide { grid-template-columns: 1fr; }
    }
</style>
<header class="sav-header">
    <h1>Assistance SAV</h1>
    <p>Dépannage et garantie de vos appareils</p>
</header>
""")
history = st.session_state.setdefault('sav_history', st.session_state.get('histories', {}).get('SAV', []))

submission = st.chat_input(
    'Décrivez le problème ou saisissez le numéro de série',
    accept_file=True,
    file_type=['png', 'jpg', 'jpeg', 'webp'],
)

if not history and submission is None:
    st.html("""
    <div class="sav-guide">
        <div>
            <strong>Dépanner un appareil</strong>
            <p>Indiquez le modèle et le symptôme. Joignez une photo du voyant ou de l’écran avec le trombone.</p>
        </div>
        <div>
            <strong>Vérifier une garantie</strong>
            <p>Saisissez le numéro de série de votre appareil, par exemple PX200-0042.</p>
        </div>
    </div>
    """)


def show_message(role, text, photo=None):
    with st.chat_message(role):
        st.write(text)
        if photo:
            st.image(photo, caption='Photo jointe', width=360)


for message in history:
    show_message(*message)


question = 'Que signifie le pictogramme sur la photo ?'
photo = None
image_ref = ''
upload_path = None

if submission is not None:
    question = submission.text.strip() or question
    if submission.files:
        try:
            with Image.open(submission.files[0]) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1600, 1600))
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                photo = buffer.getvalue()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error('Impossible de lire cette image. Essaie un fichier PNG, JPEG ou WebP.')
            st.stop()
        image_ref = f'upload_{uuid4().hex}'
        upload_path = ROOT / 'data' / f'{image_ref}.png'

    user_message = ('user', question, photo)
    history.append(user_message)
    show_message(*user_message)

    try:
        if upload_path is not None:
            upload_path.write_bytes(photo)
        with st.spinner('Traitement de la demande SAV…', show_time=True):
            result = asyncio.run(answer_request(question, image_ref))
        answer = result['answer']
    except TimeoutError:
        answer = f'La demande a dépassé {REQUEST_TIMEOUT} secondes. Réessaie après avoir vérifié le terminal du serveur SAV.'
    except Exception as error:
        logger.warning('Échec de la demande SAV (%s)', type(error).__name__)
        answer = 'La demande SAV a échoué. Vérifie le terminal du serveur SAV et ta configuration Groq.'
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)

    history.append(('assistant', answer))
    show_message('assistant', answer)
