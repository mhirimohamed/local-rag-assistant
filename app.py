
# # 🧠 Local RAG Assistant — Privacy‑First AI Chatbot with Gradio

# A fully local, production‑ready **RAG (Retrieval‑Augmented Generation)** assistant built with **Gradio**, **Ollama**, and **Qdrant**.  
# Upload your documents, generate a knowledge base, and chat with your own data — **without sending anything to the cloud**.

# Perfect for **insurance**, **finance**, **government**, **healthcare**, and any domain requiring **strict data privacy**.

# ---

# ## ✨ Features

# - 🔒 **100% local inference** using Ollama  
# - 📚 **RAG pipeline** with Qdrant vector search  
# - 📁 Upload PDF/TXT/DOCX and build a knowledge base  
# - 🎛️ Adjustable **temperature**, **seed**, and **system prompt**  
# - 🔍 Source filtering for precise answers  
# - 💬 Clean Gradio UI with chat history  
# - ⚡ Fast ingestion + deterministic responses  
# - 🧩 Modular architecture (UI / LLM / RAG / Handlers)

import gradio as gr

from config import (
    APP_TITLE, APP_DESC, CUSTOM_CSS, DEFAULT_TOPK
)
from llm.ollama_chat import get_local_models
from rag.ingestion import ingest_into_qdrant
from ui.handlers import (
    on_send_submit_rag, on_clear_chat
)

def build_app():
    with gr.Blocks(fill_height=True) as demo:

        gr.Markdown(f"""<div style="line-height: 1.2;"><span style="color:black;font-size: 36px;font-weight:700;">{APP_TITLE}</span>
                    <br><span style="color:black;font-size: 24px;font-weight:500;">{APP_DESC}</span></div>""")
        with gr.Row():
            with gr.Column(scale=3):
                system_prompt = gr.Textbox(
                    label="Personnaliser le système",
                    placeholder="Vous êtes un assistant Chatbot : répondez de façon concise et précise.",
                    lines=3,
                )

                initial_models = get_local_models()
                model_dropdown = gr.Dropdown(
                    label="Modèle de langage",
                    choices=initial_models,
                    value=(initial_models[0] if initial_models else None),
                    allow_custom_value=True,
                    info="Les modèles sont détectés localement.",
                )

                with gr.Row():
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Température")
                    seed = gr.Number(value=0, precision=0, label="Graine")

                gr.Markdown(
                    "* **Graine** : garantit les mêmes résultats.\n"
                    "* **Température** : 0 = très déterministe, 1 = très créatif."
                )

                # --- RAG UI ---
                rag_enabled = gr.Checkbox(label="Activer le RAG", value=False)
                rag_topk = gr.Slider(1, 20, value=DEFAULT_TOPK, step=1, label="Top‑K chunks",visible=False)
                rag_files = gr.Files(
                    label="Documents (PDF/TXT/DOCX) à ingérer",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"],
                    visible=False,
                    interactive=True,
                )
                build_kb_btn = gr.Button("Générer la Base de Connaissances", variant="primary", visible=False)
                kb_status = gr.Markdown("", visible=False)
                sources_state = gr.State(value=[])
                sources_filter = gr.CheckboxGroup(choices=[], label="Filtrer par source (optionnel)", visible=False)

                def _toggle_rag_files(is_rag_enabled: bool):
                    return gr.update(visible=bool(is_rag_enabled))

                rag_enabled.change(fn=_toggle_rag_files, inputs=[rag_enabled], outputs=[rag_topk], queue=False)
                rag_enabled.change(fn=_toggle_rag_files, inputs=[rag_enabled], outputs=[rag_files], queue=False)
                rag_enabled.change(fn=_toggle_rag_files, inputs=[rag_enabled], outputs=[build_kb_btn], queue=False)
                rag_enabled.change(fn=_toggle_rag_files, inputs=[rag_enabled], outputs=[kb_status], queue=False)
                rag_enabled.change(fn=_toggle_rag_files, inputs=[rag_enabled], outputs=[sources_filter], queue=False)

                def _ingest_and_update(files):
                    status, sources = ingest_into_qdrant(files)
                    return status, sources, gr.update(choices=sources, value=sources)

                build_kb_btn.click(
                    fn=_ingest_and_update,
                    inputs=[rag_files],
                    outputs=[kb_status, sources_state, sources_filter],
                )

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=420, show_label=False)
                prompt_box = gr.Textbox(placeholder="Posez-moi vos questions ...", lines=3)

                with gr.Row():
                    send_btn = gr.Button("Envoyer", variant="primary")
                    clear_btn = gr.Button("Nouvelle discussion")

                send_btn.click(
                    fn=on_send_submit_rag,
                    inputs=[prompt_box, chatbot, system_prompt, model_dropdown, temperature, seed,
                            rag_enabled, rag_topk, sources_filter],
                    outputs=[prompt_box, chatbot],
                )
                prompt_box.submit(
                    fn=on_send_submit_rag,
                    inputs=[prompt_box, chatbot, system_prompt, model_dropdown, temperature, seed,
                            rag_enabled, rag_topk, sources_filter],
                    outputs=[prompt_box, chatbot],
                )

                clear_btn.click(fn=on_clear_chat, outputs=[chatbot, prompt_box])

    return demo

if __name__ == "__main__":
    demo = build_app()
    # Gradio 6: passer theme & css dans launch() )
    demo.queue(max_size=128).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
