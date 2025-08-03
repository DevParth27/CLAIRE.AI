import re
from datetime import datetime
from collections import defaultdict

def analyze_execution_logs(log_file='hackrx_execution.log'):
    """Analyze execution logs for Railway deployment monitoring"""
    
    sessions = defaultdict(dict)
    questions = defaultdict(list)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        timestamp = parts[0].split(' - ')[0]
                        log_type = parts[1] if len(parts) > 1 else ''
                        session_id = parts[2] if len(parts) > 2 else ''
                        
                        if log_type == 'SESSION_START':
                            sessions[session_id]['start_time'] = timestamp
                            sessions[session_id]['questions_count'] = parts[4].split(':')[1] if len(parts) > 4 else '0'
                            
                        elif log_type == 'QUESTION_RECEIVED':
                            question_num = parts[3]
                            question_text = parts[4] if len(parts) > 4 else ''
                            questions[session_id].append({
                                'number': question_num,
                                'text': question_text,
                                'timestamp': timestamp
                            })
                            
                        elif log_type == 'ANSWER_GENERATED':
                            question_num = parts[3]
                            time_info = parts[4] if len(parts) > 4 else ''
                            sessions[session_id][f'q{question_num}_success'] = True
                            sessions[session_id][f'q{question_num}_time'] = time_info
                            
                        elif log_type == 'SESSION_COMPLETE':
                            sessions[session_id]['end_time'] = timestamp
                            sessions[session_id]['total_time'] = parts[4] if len(parts) > 4 else ''
                            sessions[session_id]['success_rate'] = parts[5] if len(parts) > 5 else ''
    
    except FileNotFoundError:
        print("Log file not found. Check Railway deployment logs.")
        return
    
    # Enhanced Railway-friendly output
    print("\n🚀 RAILWAY DEPLOYMENT - EXECUTION LOG ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 Total Sessions: {len(sessions)}")
    
    for session_id, data in sessions.items():
        print(f"\n📋 Session: {session_id}")
        print(f"   🕐 Start: {data.get('start_time', 'N/A')}")
        print(f"   📝 Questions: {data.get('questions_count', 'N/A')}")
        print(f"   ⏱️  Total Time: {data.get('total_time', 'N/A')}")
        print(f"   ✅ Success Rate: {data.get('success_rate', 'N/A')}")
        
        if session_id in questions:
            print(f"\n   📋 QUESTIONS PROCESSED:")
            print(f"   {'-' * 50}")
            for q in questions[session_id]:
                # Clean question display
                question_text = q['text'].replace('...', '').strip()
                print(f"   {q['number']}: {question_text}")
            print(f"   {'-' * 50}")
    
    print("\n" + "=" * 60)
    print("🔍 To view live logs on Railway: railway logs --tail")

def show_latest_questions(log_file='hackrx_execution.log'):
    """Show only the latest session questions - Railway friendly"""
    
    latest_session = None
    questions = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Find latest session
        for line in reversed(lines):
            if 'SESSION_START|' in line:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    latest_session = parts[2]
                break
        
        if latest_session:
            for line in lines:
                if f'QUESTION_RECEIVED|{latest_session}|' in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 5:
                        q_num = parts[3]
                        q_text = parts[4]
                        questions.append(f"{q_num}: {q_text}")
        
        print("\n🎯 LATEST SESSION QUESTIONS")
        print("=" * 40)
        for q in questions:
            print(f"📝 {q}")
        print("=" * 40)
        
    except FileNotFoundError:
        print("❌ No log file found. Check Railway deployment.")

if __name__ == "__main__":
    print("Choose analysis type:")
    print("1. Full analysis (analyze_execution_logs)")
    print("2. Latest questions only (show_latest_questions)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        show_latest_questions()
    else:
        analyze_execution_logs()