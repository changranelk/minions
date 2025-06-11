#!/usr/bin/env python3
"""
SambaNova API Integration Test
Tests the combination of DeepSeek-V3-0324 (supervisor) + Meta-Llama-3.1-8B-Instruct (worker)
with 3M 2015 10K document analysis for 2014 revenue information.
"""

import os
import sys
import time
import re
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from minions.clients.sambanova_remote import SambanovaRemoteClient
from minions.clients.sambanova_local import SambanovaLocalClient  
from minions.minions import Minions
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None


def test_sambanova_integration():
    """Test SambaNova API integration with Minions protocol."""
    
    print("🧪 Starting SambaNova API Integration Test")
    print("=" * 50)
    
    # Test configuration
    pdf_path = "/Users/changranh/Downloads/3M_2015_10K.pdf"
    task = "What was 3M's total revenue in 2014? Please provide the exact figure in billions of dollars."
    doc_metadata = "3M Company 2015 10-K Annual Report"
    
    # Check PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    print(f"✅ PDF file found: {pdf_path}")
    
    # Check API key
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        print("❌ SAMBANOVA_API_KEY not set")
        return False
    print("✅ API key configured")
    
    # Extract PDF text
    print("\n📄 Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("❌ Failed to extract PDF text")
        return False
    print(f"✅ Extracted {len(pdf_text)} characters from PDF")
    
    # Initialize clients
    print("\n🔧 Initializing SambaNova clients...")
    try:
        # Remote client (supervisor) - DeepSeek-V3-0324
        remote_client = SambanovaRemoteClient(
            model_name="DeepSeek-V3-0324",
            api_key=api_key,
            max_tokens=8000,
            temperature=0.0
        )
        print("✅ Remote client (DeepSeek-V3-0324) initialized")
        
        # Local client (worker) - Meta-Llama-3.1-8B-Instruct
        local_client = SambanovaLocalClient(
            model_name="Meta-Llama-3.1-8B-Instruct", 
            api_key=api_key,
            max_tokens=8000,
            num_ctx=8000,
            temperature=0.0
        )
        print("✅ Local client (Meta-Llama-3.1-8B-Instruct) initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        return False

    # Initialize Minions with debug logging
    print("\n🤖 Initializing Minions protocol...")
    try:
        minions = Minions(
            remote_client=remote_client,
            local_client=local_client,
            max_rounds=3,
            enable_logging=True  # Enable detailed logging
        )
        print("✅ Minions protocol initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Minions: {e}")
        return False

    # Run analysis
    print(f"\n🔍 Testing query: {task}")
    print("⏳ Running Minions analysis...")
    
    start_time = time.time()
    
    # Create debug log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"sambanova_test_debug_{timestamp}.log"
    
    try:
        with open(log_file, 'w') as f:
            f.write("=== SambaNova Integration Test Debug Log ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Document: {doc_metadata}\n")
            f.write("=" * 50 + "\n\n")
        
        # Monkey patch the clients to log prompts and responses
        original_remote_chat = remote_client.chat
        original_local_chat = local_client.chat
        
        def log_remote_chat(messages, **kwargs):
            with open(log_file, 'a') as f:
                f.write("🔴 REMOTE CLIENT (SUPERVISOR) CALL:\n")
                f.write(f"Messages: {messages}\n")
                f.write(f"Kwargs: {kwargs}\n")
                f.write("-" * 30 + "\n")
            
            response = original_remote_chat(messages, **kwargs)
            
            with open(log_file, 'a') as f:
                f.write("🔴 REMOTE CLIENT RESPONSE:\n")
                f.write(f"Response: {response}\n")
                f.write("=" * 50 + "\n\n")
            
            return response
        
        def log_local_chat(messages, **kwargs):
            with open(log_file, 'a') as f:
                f.write("🔵 LOCAL CLIENT (WORKER) CALL:\n")
                f.write(f"Messages: {messages}\n")
                f.write(f"Kwargs: {kwargs}\n")
                f.write("-" * 30 + "\n")
            
            response = original_local_chat(messages, **kwargs)
            
            with open(log_file, 'a') as f:
                f.write("🔵 LOCAL CLIENT RESPONSE:\n")
                f.write(f"Response: {response}\n")
                f.write("=" * 50 + "\n\n")
            
            return response
        
        # Apply monkey patches
        remote_client.chat = log_remote_chat
        local_client.chat = log_local_chat
        
        result = minions(
            task=task,
            context=[pdf_text],
            doc_metadata=doc_metadata
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Analysis completed in {duration:.2f} seconds")
        print(f"📋 Debug log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        with open(log_file, 'a') as f:
            f.write(f"ERROR: {e}\n")
            f.write(f"Error type: {type(e)}\n")
            import traceback
            f.write(f"Traceback: {traceback.format_exc()}\n")
        print(f"📋 Error log saved to: {log_file}")
        return False

    # Analyze result
    print("\n📊 Analyzing results...")
    answer = result.get('generated_final_answer', 'No answer found')
    print(f"💬 Answer: {answer}")
    
    # Extract revenue number from answer
    revenue_number = None
    if answer and answer != "No answer found":
        # Look for number patterns like 31.5, $31.5, 31.5B, etc.
        patterns = [
            r'\$?(\d+\.?\d*)\s*billion',
            r'\$?(\d+\.?\d*)\s*B',
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer.lower())
            if matches:
                try:
                    revenue_number = float(matches[0])
                    break
                except ValueError:
                    continue
    
    if revenue_number is None:
        print("❌ Could not extract revenue number from response")
        print(f"Full response: {answer}")
        print(f"📋 Check debug log for detailed prompts/responses: {log_file}")
        return False
    
    print(f"💰 Extracted revenue: ${revenue_number:.1f} billion")
    
    # Validate expected range (31-32 billion based on hint)
    expected_min = 31.0
    expected_max = 32.0
    
    if expected_min <= revenue_number <= expected_max:
        print(f"✅ SUCCESS: Revenue ${revenue_number:.1f}B is within expected range (${expected_min:.1f}B - ${expected_max:.1f}B)")
        return True
    else:
        print(f"❌ FAIL: Revenue ${revenue_number:.1f}B is outside expected range (${expected_min:.1f}B - ${expected_max:.1f}B)")
        return False


if __name__ == "__main__":
    print("🚀 SambaNova API Integration Test")
    print("Testing DeepSeek-V3-0324 + Meta-Llama-3.1-8B-Instruct")
    print("Document: 3M 2015 10-K Report")
    print("Query: 2014 Revenue")
    print()
    
    success = test_sambanova_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST PASSED: SambaNova integration working correctly")
    else:
        print("💥 TEST FAILED: Issues detected with SambaNova integration") 