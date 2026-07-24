"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import cli, load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time
    response_text = response.choices[0].message.content
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    model: str = OPENAI_MINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    return call_openai(
        prompt=prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )



# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    call_gpt4o = call_openai(prompt)
    call_mini = call_openai_mini(prompt)
    gpt4o_response, gpt4o_latency = call_gpt4o
    mini_response, mini_latency = call_mini
    gpt4o_cost_estimate = (
        len(gpt4o_response.split()) / 0.75 / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )
    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate
    }



# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time
    response_text = response.choices[0].message.content
    return response_text, latency



# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try :
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        return max(1, len(text) // 4)



# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    input_price_per_1k = pricing.get("input", PRICING_PER_1K_TOKENS["gpt-4o"]["input"])
    output_price_per_1k = pricing.get("output", PRICING_PER_1K_TOKENS["gpt-4o"]["output"])
    input_cost = input_tokens / 1000.0 * input_price_per_1k
    output_cost = output_tokens / 1000.0 * output_price_per_1k
    total_cost = input_cost + output_cost
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(total_cost),
    } 



# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:    
    from openai import OpenAI

    history: list[dict[str, str]] = []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while True:
        user_msg = input()
        if user_msg is None:
            break
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = history + [{"role": "user", "content": user_msg}]

        # Stream the response from the API
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True
        )

        reply_parts: list[str] = []
        for chunk in stream:
            delta = None
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                print(delta, end="", flush=True)
                reply_parts.append(delta)
        print()

        reply = "".join(reply_parts)

        # Append the turn (user + assistant) and trim to last 3 turns (6 messages)
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ])
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:

    if get_input is None:
        get_input = input

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history: list[dict[str, str]] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        try:
            user_msg = get_input()
        except StopIteration:
            break

        if user_msg is None:
            break
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]
        )

        # Get a streaming iterator, with retries for transient failures
        stream = retry_with_backoff(lambda: client.chat.completions.create(model=OPENAI_MODEL, messages=messages, stream=True))

        reply_parts: list[str] = []
        for chunk in stream:
            delta = None
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                print(delta, end="", flush=True)
                reply_parts.append(delta)
        print()

        reply = "".join(reply_parts)

        # Update stats
        input_tokens = count_tokens(user_msg, OPENAI_MODEL)
        output_tokens = count_tokens(reply, OPENAI_MODEL)
        costs = estimate_cost(user_msg, reply, OPENAI_MODEL)

        total_tokens += input_tokens + output_tokens
        total_cost += costs.get("total_cost", 0.0)

        # Update history and turns
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ])
        history = history[-6:]

        num_turns += 1

    return {
        "num_turns": int(num_turns),
        "total_tokens": int(total_tokens),
        "total_cost": float(total_cost),
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:

    results: list[dict] = []
    for p in prompts:
        result = compare_models(p)
        entry = dict(result)  # copy to avoid mutating the original
        entry["prompt"] = p
        results.append(entry)
    return results


def format_comparison_table(results: list[dict]) -> str:
    def _truncate(text: Any, width: int = 40) -> str:
        s = "" if text is None else str(text)
        s = " ".join(s.splitlines())
        s = s.strip()
        return s if len(s) <= width else s[: width - 3] + "..."

    w_prompt, w_gpt, w_mini, w_lat = 40, 40, 40, 12
    header = (
        f"{'Prompt':{w_prompt}} | "
        f"{'GPT-4o Response':{w_gpt}} | "
        f"{'Mini Response':{w_mini}} | "
        f"{'GPT-4o Latency':{w_lat}} | "
        f"{'Mini Latency':{w_lat}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        p = _truncate(r.get("prompt", ""), w_prompt)
        g = _truncate(r.get("gpt4o_response", ""), w_gpt)
        m = _truncate(r.get("mini_response", ""), w_mini)
        gl = _truncate(r.get("gpt4o_latency", ""), w_lat)
        ml = _truncate(r.get("mini_latency", ""), w_lat)
        lines.append(
            f"{p:{w_prompt}} | {g:{w_gpt}} | {m:{w_mini}} | {gl:{w_lat}} | {ml:{w_lat}}"
        )

    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
