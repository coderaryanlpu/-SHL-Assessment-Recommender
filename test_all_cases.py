import os
import re
import glob
from agent import run_agent

def parse_markdown_trace(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract user utterances
    # They look like:
    # **User**
    # 
    # > The pool consists of CXOs...
    
    user_turns = []
    # A simple regex to find the blockquote after **User**
    blocks = content.split('**User**')
    for block in blocks[1:]:
        # Find the text starting with > and ending at next double newline
        match = re.search(r'>\s*(.*?)(?:\n\n|\n\*\*|\Z)', block, re.DOTALL)
        if match:
            text = match.group(1).replace('\n> ', ' ').replace('\n>', ' ').strip()
            user_turns.append(text)
    return user_turns

def run_trace(name, user_turns):
    print(f"\n{'='*50}\nRunning Trace: {name}\n{'='*50}")
    messages = []
    
    for i, user_text in enumerate(user_turns):
        print(f"\n[Turn {i+1}] User: {user_text}")
        messages.append({"role": "user", "content": user_text})
        
        # Call agent
        result = run_agent(messages)
        
        reply = result.get("reply", "")
        recs = result.get("recommendations", [])
        eoc = result.get("end_of_conversation", False)
        
        print(f"Agent: {reply}")
        if recs:
            print(f"  -> Recommended {len(recs)} items:")
            for r in recs:
                print(f"     - {r['name']} ({r['test_type']})")
        else:
            print("  -> No recommendations yet.")
            
        print(f"  -> End of Conversation: {eoc}")
        
        # Add assistant reply to history
        messages.append({"role": "assistant", "content": reply})
        
        if eoc:
            print("\n*** Agent ended conversation early ***")
            break

if __name__ == "__main__":
    trace_files = sorted(glob.glob("GenAI_SampleConversations/C*.md"), key=lambda x: int(re.search(r'C(\d+)', x).group(1)))
    print(f"Found {len(trace_files)} test traces.")
    
    for filepath in trace_files:
        name = os.path.basename(filepath)
        user_turns = parse_markdown_trace(filepath)
        if user_turns:
            run_trace(name, user_turns)
        else:
            print(f"Could not parse user turns from {name}")
