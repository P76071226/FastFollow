from dotenv import load_dotenv
import re
import os
import dspy
import gradio as gr
from typing import List, Dict, Optional, Tuple

load_dotenv()
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# ----------------- DSPy signatures -----------------
class MainAnswerSig(dspy.Signature):
    """Answer the user's question clearly and concisely (2–5 sentences)."""
    question = dspy.InputField()
    answer   = dspy.OutputField()

class ProposeFollowupsSig(dspy.Signature):
    """Propose 3–5 specific, non-overlapping follow-up questions as a numbered list."""
    context   = dspy.InputField()
    followups = dspy.OutputField()

class FollowupAnswerSig(dspy.Signature):
    """Answer the follow-up question concisely (2–4 sentences)."""
    base_question     = dspy.InputField()
    followup_question = dspy.InputField()
    answer            = dspy.OutputField()

# ----------------- DSPy modules -----------------
class MainAnswerer(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(MainAnswerSig)
    def forward(self, question: str) -> str: return self.p(question=question).answer

class FollowupProposer(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(ProposeFollowupsSig)
    @staticmethod
    def _parse(text: str, k: int) -> List[str]:
        items, seen = [], set()
        for raw in text.splitlines():
            m = re.match(r'^\s*(?:\d+\.\s*|[-•]\s*)?(.*\S)\s*$', raw)
            if not m: continue
            t = m.group(1).strip()
            if not t: continue
            if t.lower().startswith(("follow-ups", "followups", "questions:")):
                continue
            if t not in seen:
                seen.add(t); items.append(t)
            if len(items) >= k: break
        return items
    def forward(self, context: str, k: int = 4) -> List[str]:
        raw = self.p(context=context).followups or ""
        return self._parse(raw, k)

class FollowupAnswerer(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(FollowupAnswerSig)
    def forward(self, base_q: str, fu_q: str) -> str:
        return self.p(base_question=base_q, followup_question=fu_q).answer

# ----------------- One-layer flow -----------------
class OneLayerFastFollow:
    def __init__(self, k_followups: int = 4):
        self.k = k_followups
        self.main = MainAnswerer()
        self.propose = FollowupProposer()
        self.answer_fu = FollowupAnswerer()
        self.base: Optional[str] = None
        self.menu: List[str] = []
        self.answers: Dict[int, str] = {}

    def _menu_text(self, title: str = "Follow-ups") -> str:
        if not self.menu: return ""
        lines = [f"{i+1}. {q}" for i, q in enumerate(self.menu)]
        return f"\n\n{title}:\n" + "\n".join(lines)

    def ask(self, question: str) -> Tuple[str, List[str]]:
        self.base = question
        main_answer = self.main(question)
        ctx = f"User question: {question}\nAnswer: {main_answer}"
        self.menu = self.propose(ctx, self.k)
        self.answers.clear()
        for idx, fu in enumerate(self.menu, start=1):
            self.answers[idx] = self.answer_fu(question, fu)
        visible = f"{main_answer}{self._menu_text()}"
        return visible, self.menu

    def peek_immediate(self, choice_idx: int) -> str:
        if not (1 <= choice_idx <= len(self.menu)):
            return f"Please choose a number between 1 and {len(self.menu)}."
        return self.answers.get(choice_idx, "Sorry, no cached answer found.")

    def rotate_next_layer(self, choice_idx: int) -> List[str]:
        selected_fu = self.menu[choice_idx - 1]
        immediate = self.answers.get(choice_idx, "")
        self.base = selected_fu
        ctx = f"Base: {selected_fu}\nAnswer: {immediate}"
        next_menu = self.propose(ctx, self.k)
        self.menu = next_menu
        self.answers.clear()
        for i, fu in enumerate(self.menu, start=1):
            self.answers[i] = self.answer_fu(selected_fu, fu)
        return self.menu

# ----------------- Gradio wiring -----------------
MAX_BTNS = 4
flow = OneLayerFastFollow(k_followups=MAX_BTNS)

def _buttons_update(menu: List[str]):
    # Use gr.update(...) (not Button.update)
    updates = []
    for i in range(MAX_BTNS):
        if i < len(menu):
            updates.append(gr.update(value=menu[i], visible=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates  # [b1, b2, b3, b4]

def submit_or_choose(user_text, chat_history):
    """
    Generator so we can yield the immediate answer first.
    - If user_text is digits: treat as follow-up selection (immediate -> yield; then rotate -> yield).
    - Otherwise: treat as new question (ask -> yield once).
    """
    text = (user_text or "").strip()
    # numeric -> follow-up selection
    if text.isdigit():
        if not flow.menu:
            chat_history = chat_history + [(text, "No follow-ups available. Ask a new question.")]
            # No button changes yet
            yield chat_history, *[gr.update() for _ in range(MAX_BTNS)]
            return
        idx = int(text)
        # 1) immediate cached answer
        immediate = flow.peek_immediate(idx)
        chat_history = chat_history + [(text, immediate)]
        yield chat_history, *[gr.update() for _ in range(MAX_BTNS)]

        # 2) rotate to next layer
        next_menu = flow.rotate_next_layer(idx)
        b1, b2, b3, b4 = _buttons_update(next_menu)
        yield chat_history, b1, b2, b3, b4
        return

    # non-numeric -> new question
    if not text:
        yield chat_history, *[gr.update() for _ in range(MAX_BTNS)]
        return

    visible, menu = flow.ask(text)
    chat_history = chat_history + [(text, visible)]
    b1, b2, b3, b4 = _buttons_update(menu)
    yield chat_history, b1, b2, b3, b4

def choose_btn(idx, chat_history):
    """Button click handler: same two-phase behavior."""
    # 1) immediate
    immediate = flow.peek_immediate(idx)
    chat_history = chat_history + [(str(idx), immediate)]
    yield chat_history, *[gr.update() for _ in range(MAX_BTNS)]
    # 2) next layer
    next_menu = flow.rotate_next_layer(idx)
    b1, b2, b3, b4 = _buttons_update(next_menu)
    yield chat_history, b1, b2, b3, b4

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("### Fast Follow-ups (DSPy, one-layer cache)\nType a question, or type a **number** to pick a follow-up.")

    chatbot = gr.Chatbot(height=420)
    txt = gr.Textbox(placeholder="Ask a question or type a number…", autofocus=True)

    with gr.Row():
        btn1 = gr.Button(visible=False)
        btn2 = gr.Button(visible=False)
        btn3 = gr.Button(visible=False)
        btn4 = gr.Button(visible=False)

    # Text submit (generator): immediate for numeric, one-shot for new question
    txt.submit(submit_or_choose, [txt, chatbot], [chatbot, btn1, btn2, btn3, btn4], show_progress="minimal").then(
        lambda: "", None, txt  # clear box
    )

    # Button clicks (generators)
    btn1.click(choose_btn, [gr.State(1), chatbot], [chatbot, btn1, btn2, btn3, btn4], show_progress="minimal")
    btn2.click(choose_btn, [gr.State(2), chatbot], [chatbot, btn1, btn2, btn3, btn4], show_progress="minimal")
    btn3.click(choose_btn, [gr.State(3), chatbot], [chatbot, btn1, btn2, btn3, btn4], show_progress="minimal")
    btn4.click(choose_btn, [gr.State(4), chatbot], [chatbot, btn1, btn2, btn3, btn4], show_progress="minimal")

demo.launch(debug=True)

