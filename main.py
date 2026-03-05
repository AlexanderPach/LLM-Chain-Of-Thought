import subprocess
import re
import os
from google import genai

CLINGO_PATH = "clingo"
OUTPUT_FILE = "warehouse_output.lp"
MAX_ITERS = 8
BACKTICKS = "```"

# gemini setup
_api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=_api_key)


def query_gemini(prompt):
    """Send a prompt to Gemini and return the text response."""
    try:
        resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return resp.text
    except Exception as e:
        raise Exception(f"Gemini API error: {e}")


def extract_asp_code(response_text):
    """Pull out the clingo/ASP code from the LLM's markdown response."""
    # try to grab code from a fenced block first
    m = re.search(r"```(?:clingo|lp|asp)?\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # sometimes the model just says "Answer: ..." with no fences
    m = re.search(r'Answer:\s*(.*?)\n', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # worst case just return everything
    return response_text.strip()


def run_clingo(filepath):
    """Run clingo on the given .lp file. Returns (success, output_or_error)."""
    try:
        result = subprocess.run(
            [CLINGO_PATH, filepath],
            capture_output=True, text=True, timeout=15
        )
        stdout = result.stdout
        stderr = result.stderr

        if "UNSATISFIABLE" in stdout:
            return False, "UNSATISFIABLE - check constraints"
        elif result.returncode in (10, 30):  # SAT or SAT/UNSAT
            return True, stdout
        else:
            return False, f"Error:\n{stderr}\n{stdout}"

    except subprocess.TimeoutExpired:
        return False, "Timed out (grounding too large?)"
    except Exception as e:
        return False, f"Failed to run clingo: {e}"


def iterative_solve(task_prompt, system_prompt):
    prompt = f"{system_prompt}\n\nHere is the main problem to solve:\n{task_prompt}"

    for i in range(1, MAX_ITERS + 1):
        print(f"\n{'='*40}")
        print(f"  Iteration {i}/{MAX_ITERS}")

        # ask gemini
        print("Querying Gemini...")
        try:
            raw_response = query_gemini(prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            continue

        # pull out the ASP code and write it to disk
        code = extract_asp_code(raw_response)
        with open(OUTPUT_FILE, "w") as f:
            f.write(code)
        print(f"Wrote code to {OUTPUT_FILE}")

        # verify
        print("Running clingo...")
        ok, output = run_clingo(OUTPUT_FILE)

        if ok:
            print("\nFound valid answer set!")
            print(output[:500])
            return True

        # didn't work - show what went wrong and loop back
        print("Invalid or UNSAT")
        print(output[:800])

        # build feedback prompt for next iteration
        prompt = (
            f"Your previous code had this error when run in Clingo:\n"
            f"{BACKTICKS}\n{output}\n{BACKTICKS}\n"
            f"Analyze the error, fix the logic/syntax, and output the "
            f"corrected complete code in a {BACKTICKS}clingo block."
        )

    print(f"\nNo valid solution after {MAX_ITERS} iterations.")
    return False


if __name__ == "__main__":
    print("Prompting LLM...\n")

    # (N-Queens)
    # This teaches the model what good ASP structure looks like (dataset)
    system_prompt = f"""You are an expert ASP (Answer Set Programming) programmer.
Always put your final code inside a {BACKTICKS}clingo block.

Here's an example of solving a CSP step-by-step with Chain-of-Thought:

--- N-Queens (n=4) ---
Query: Place N queens on an NxN board so no two attack each other.
Constraints: 1 queen per row, 1 per column, at most 1 per diagonal.

CoT:
1. Define grid domain (rows/cols 1..N)
2. Choice rule: exactly one queen per row
3. Constraint: no shared columns
4. Constraint: no shared diagonals (|R1-R2| == |C1-C2|)

Solution:
{BACKTICKS}clingo
#const n = 4.
row(1..n). col(1..n).
{{ queen(R,C) : col(C) }} == 1 :- row(R).
:- queen(R1,C), queen(R2,C), R1 != R2.
:- queen(R1,C1), queen(R2,C2), R1 != R2, |R1 - R2| == |C1 - C2|.
#show queen/2.
{BACKTICKS}
"""

    # the actual warehouse problem we want to solve
    warehouse_prompt = f"""Automated Warehouse Scenario:
Grid: 3x3. Two robots: r1 and r2.
- r1 starts at (1,1), needs to reach shelf at (1,3).
- r2 starts at (3,3), needs to reach shelf at (3,1).
- Max time steps: 4 (t=0 to t=4).

Constraints:
1. Robots can move up/down/left/right or wait each step.
2. Can't move outside the grid.
3. No vertex collision (two robots same cell same time).
4. No edge collision (two robots swapping adjacent cells).
5. Both robots must be at their destinations at t=4.

Use this ASP planning structure:
- Base facts: robot(r1). robot(r2). time(0..4). etc.
- Initial positions at t=0.
- Generate exactly one action per robot per step (t=0..3).
- Effects: action at t determines position at t+1.
- Integrity constraints for bounds, collisions, goals.

Write the complete Clingo program. Use #show pos/4. to display
robot positions at each timestep.
Give a brief CoT explanation before the {BACKTICKS}clingo block.
"""

iterative_solve(warehouse_prompt, system_prompt)