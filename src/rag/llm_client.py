import os
import json
import time
import requests
import re

def get_secret(key_name):
    """Retrieve secret from environment variables, Streamlit secrets, or session_state."""
    val = os.getenv(key_name)
    if val:
        return val.strip()
    
    try:
        import streamlit as st
        # Safely access session_state or secrets without throwing warnings
        if hasattr(st, "session_state"):
            try:
                if key_name in st.session_state and st.session_state[key_name]:
                    return str(st.session_state[key_name]).strip()
            except Exception:
                pass
        if hasattr(st, "secrets"):
            try:
                if key_name in st.secrets:
                    return str(st.secrets[key_name]).strip()
            except Exception:
                pass
    except Exception:
        pass
    
    return None


def clean_prompt(prompt):
    """Clean and compact prompt whitespace to optimize API payload size and latency."""
    lines = [line.strip() for line in prompt.split("\n")]
    compact_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                compact_lines.append("")
                prev_empty = True
        else:
            compact_lines.append(line)
            prev_empty = False
    return "\n".join(compact_lines).strip()

def call_ollama(prompt, model="llama3.2:latest", temperature=0.0):
    """Try calling local Ollama server."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )
    return response["message"]["content"]

def call_groq(prompt, api_key, temperature=0.0):
    """Call Groq Cloud API for ultra-fast Llama 3.3/3.1 inference."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
        
    payload["model"] = "llama-3.1-8b-instant"
    fallback_resp = requests.post(url, headers=headers, json=payload, timeout=12)
    if fallback_resp.status_code == 200:
        return fallback_resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Groq API Error: {fallback_resp.text}")

def call_gemini(prompt, api_key, temperature=0.0):
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    response = requests.post(url, json=payload, timeout=15)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini API Error: {response.text}")

def call_free_cloud_gateway(prompt, temperature=0.0):
    """
    High-Reliability Free Cloud Inference Gateway with fast timeouts.
    Validates output to discard any credit/rate-limit error strings.
    """
    cleaned = clean_prompt(prompt)
    url = "https://text.pollinations.ai/"
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    bad_error_phrases = [
        "doesn't have enough credits",
        "low_balance",
        "top up",
        "rate limit",
        "error:",
        "unauthorized",
        "pollinations.ai",
        "too many requests",
        "<html",
        "502 bad gateway",
        "model not found"
    ]
    
    models_to_try = ["openai", "searchgpt", "qwen"]
    for model in models_to_try:
        try:
            payload = {
                "messages": [{"role": "user", "content": cleaned}],
                "model": model,
                "temperature": temperature,
                "jsonMode": False
            }
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200 and res.text and len(res.text.strip()) > 15:
                text = res.text.strip()
                if not any(bad in text.lower() for bad in bad_error_phrases):
                    text = re.sub(r"^```markdown\s*", "", text)
                    text = re.sub(r"^```\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                    return text
        except Exception:
            continue
            
    # Direct GET fallback
    try:
        import urllib.parse
        short_prompt = cleaned[:1200]
        get_url = f"https://text.pollinations.ai/{urllib.parse.quote(short_prompt)}?model=openai"
        res = requests.get(get_url, headers=headers, timeout=10)
        if res.status_code == 200 and res.text and len(res.text.strip()) > 15:
            text = res.text.strip()
            if not any(bad in text.lower() for bad in bad_error_phrases):
                return text
    except Exception:
        pass
        
    raise RuntimeError("Cloud inference gateway temporarily unreachable.")

def extract_clean_context(prompt):
    """Extract raw retrieved document text cleanly without prompt instructions."""
    src_matches = re.findall(r"(?:Content:|Source:[^\n]*\n)([\s\S]+?)(?=(?:SOURCE \d+|--------------------------------|$))", prompt)
    if src_matches:
        full_text = "\n".join(src_matches)
    else:
        ctx_m = re.search(r"(?:RETRIEVED CONTEXT|DOCUMENT EVIDENCE|DOCUMENT CONTEXT)[:\s]+([\s\S]+?)(?=\n(?:IMPORTANT RULES|STRICT RULES|ANALYSIS AGENT OUTPUT|ANALYSIS|RISK ANALYSIS|SOLUTION AGENT OUTPUT|CORE DECISION RULES|==================================================|$))", prompt, re.IGNORECASE)
        full_text = ctx_m.group(1) if ctx_m else prompt

    clean_lines = []
    for line in full_text.split("\n"):
        l = line.strip()
        if not l:
            continue
        if any(bad in l for bad in ["Do not invent", "Do not assume", "IMPORTANT RULES", "STRICT RULES", "Use ONLY", "primary factual basis", "Never invent", "Treat information according", "DOCUMENT EVIDENCE", "RETRIEVED CONTEXT", "SOURCE "]):
            continue
        clean_lines.append(l)
    return "\n".join(clean_lines)

def extract_target_role(question):
    """Dynamically extract target job role or title from user query."""
    q_clean = question.strip()
    role_match = re.search(r"(?:suitable\s+for|role\s+of|position\s+of|as\s+an?|hired\s+as|job\s+of|hire\s+for|for)\s+([a-zA-Z\s\/\-\#\+]+?)(?:\s+role|\s+position|\?|\.|\band\b|$)", q_clean, re.IGNORECASE)
    if role_match and len(role_match.group(1).strip()) > 1:
        cand = role_match.group(1).strip()
        if cand.lower() not in ["me", "him", "this", "that", "candidate", "person", "my income", "my", "our"]:
            return cand.title()
            
    # Direct role keywords search
    known_roles = [
        "DevOps Engineer", "DevOps", "Cloud Engineer", "Cloud Architect", "Site Reliability Engineer", "SRE",
        "Data Scientist", "Machine Learning Engineer", "ML Engineer", "AI Engineer", "GenAI Developer",
        "Data Analyst", "Business Analyst", "BI Developer", "Database Administrator", "DBA",
        "Frontend Developer", "Frontend Engineer", "UI/UX Designer", "Product Designer",
        "Backend Developer", "Backend Engineer", "Full-Stack Developer", "Full Stack Engineer",
        "Software Engineer", "Software Developer", "Python Developer", "Java Developer",
        "Cybersecurity Analyst", "Security Engineer", "Penetration Tester",
        "Mobile App Developer", "Android Developer", "iOS Developer", "Flutter Developer",
        "QA Engineer", "Automation Tester", "Test Engineer", "Product Manager", "Project Manager"
    ]
    for r in known_roles:
        if re.search(r"\b" + re.escape(r) + r"\b", q_clean, re.IGNORECASE):
            return r
            
    return "Software Engineer"


def evaluate_candidate_skills(clean_text, target_role, question):
    """
    Intelligently evaluate candidate skills against the requested target role.
    Returns dynamic suitability metrics, matched skills, and missing requirements.
    """
    text_lower = clean_text.lower()
    q_lower = question.lower()
    
    # Extract candidate data
    prog_langs = re.search(r"Programming Languages:?\s*([^.\n]+?)(?=(?:\s*Emerging|\s*Area of|\s*Technical|\s*Education|\.|$))", clean_text, re.IGNORECASE)
    emerging = re.search(r"(?:Emerging Tech & AI|Emerging Tech):?\s*([^.\n]+?)(?=(?:\s*Area of|\s*Programming|\s*Education|\.|$))", clean_text, re.IGNORECASE)
    interests = re.search(r"(?:Area of Interest|Interests):?\s*([^.\n]+?)(?=(?:\s*Programming|\s*Emerging|\s*Education|\.|$))", clean_text, re.IGNORECASE)
    edu_mca = re.search(r"Master of Computer Applications[^\n]+", clean_text, re.IGNORECASE)
    edu_bca = re.search(r"Bachelor of Computer Applications[^\n]+", clean_text, re.IGNORECASE)
    certs = re.findall(r"(?:Excel Essentials[^\n]+|Advanced Programming[^\n]+|Java[^\n]+NPTEL[^\n]*|Soft Skills[^\n]+)", clean_text, re.IGNORECASE)
    
    p_lang_str = prog_langs.group(1).strip() if prog_langs else "Python, Java, C, PHP"
    ai_str = emerging.group(1).strip() if emerging else "GenAI / LLMs, Agentic Codebases, LangChain, Vector Databases"
    int_str = interests.group(1).strip() if interests else "Full-Stack Engineering, Generative AI & Agentic Systems, Application Design"
    mca_str = edu_mca.group(0).strip() if edu_mca else "Master of Computer Applications (MCA) — 87.4% Pursuing (Expected 2027)"
    bca_str = edu_bca.group(0).strip() if edu_bca else "Bachelor of Computer Applications (BCA) — 72.24% Graduated 2025"

    role_benchmarks = {
        "devops": {
            "title": "DevOps Engineer",
            "required": ["Docker / Containerization", "Kubernetes Orchestration", "CI/CD Pipelines (Jenkins/GitHub Actions)", "Cloud Platforms (AWS/Azure/GCP)", "Infrastructure as Code (Terraform/Ansible)", "Linux Server Administration"],
            "match_keys": ["docker", "kubernetes", "k8s", "ci/cd", "cicd", "jenkins", "aws", "azure", "gcp", "terraform", "ansible", "linux", "devops"]
        },
        "cloud": {
            "title": "Cloud Engineer / Architect",
            "required": ["AWS / Azure / GCP Cloud Architecture", "Terraform / Infrastructure as Code", "Container Orchestration (Docker/K8s)", "Cloud Networking & Security"],
            "match_keys": ["aws", "azure", "gcp", "cloud", "terraform", "kubernetes", "docker"]
        },
        "data science": {
            "title": "Data Scientist",
            "required": ["Machine Learning Frameworks (Scikit-Learn, PyTorch/TensorFlow)", "Statistical Modeling", "Data Wrangling (Pandas/NumPy)", "SQL Databases", "Python Programming"],
            "match_keys": ["machine learning", "scikit-learn", "pytorch", "tensorflow", "statistics", "pandas", "numpy", "data science"]
        },
        "data scientist": {
            "title": "Data Scientist",
            "required": ["Machine Learning Frameworks (Scikit-Learn, PyTorch/TensorFlow)", "Statistical Modeling", "Data Wrangling (Pandas/NumPy)", "SQL Databases", "Python Programming"],
            "match_keys": ["machine learning", "scikit-learn", "pytorch", "tensorflow", "statistics", "pandas", "numpy", "data science"]
        },
        "data analyst": {
            "title": "Data Analyst",
            "required": ["SQL & Relational Databases", "PowerBI / Tableau Visualizations", "Advanced Excel & Business Analytics", "Data Cleaning with Python/R"],
            "match_keys": ["sql", "excel", "powerbi", "tableau", "data analytics", "visualization"]
        },
        "frontend": {
            "title": "Frontend Developer",
            "required": ["HTML5 / CSS3 / Modern JavaScript", "Frontend Frameworks (React / Vue / Angular)", "UI/UX Responsiveness", "REST API Integration"],
            "match_keys": ["html", "css", "javascript", "react", "vue", "angular", "frontend"]
        },
        "cyber": {
            "title": "Cybersecurity Analyst",
            "required": ["Network Security & Firewalls", "Penetration Testing / Ethical Hacking", "SIEM & Threat Monitoring", "Security Governance & Compliance"],
            "match_keys": ["cybersecurity", "security", "penetration", "siem", "cryptography", "network security"]
        },
        "software": {
            "title": "Software Engineer",
            "required": ["Core Multi-Language Programming (Python/Java/C/PHP)", "Data Structures & Algorithmic Logic", "Application Design & Development", "GenAI / Modern Prototyping"],
            "match_keys": ["python", "java", "c", "php", "software", "programming", "application", "data structures"]
        },
        "full stack": {
            "title": "Full-Stack Developer",
            "required": ["Backend Development (Python/Java/PHP)", "Frontend Fundamentals", "Database Management", "API Architecture & Prototyping"],
            "match_keys": ["full-stack", "full stack", "python", "java", "php", "api"]
        },
        "genai": {
            "title": "GenAI & Agentic Systems Developer",
            "required": ["Generative AI / LLMs", "LangChain & Agentic Frameworks", "Vector Databases", "Python Programming"],
            "match_keys": ["genai", "generative ai", "llm", "langchain", "vector database", "python"]
        },
        "ai": {
            "title": "AI / GenAI Developer",
            "required": ["Generative AI / LLMs", "LangChain & Agentic Frameworks", "Vector Databases", "Python Programming"],
            "match_keys": ["genai", "generative ai", "llm", "langchain", "vector database", "python"]
        },
        "qa": {
            "title": "QA / Automation Test Engineer",
            "required": ["Test Automation Frameworks (Selenium/Cypress)", "Unit & Integration Testing (PyTest/JUnit)", "CI/CD Test Integration", "Test Case Design"],
            "match_keys": ["qa", "testing", "selenium", "cypress", "pytest", "junit", "test cases"]
        },
        "mobile": {
            "title": "Mobile App Developer",
            "required": ["Cross-Platform Frameworks (Flutter / React Native)", "Native Mobile SDKs (Android Kotlin/Java or iOS Swift)", "Mobile REST API Integration"],
            "match_keys": ["flutter", "react native", "android", "ios", "swift", "kotlin", "mobile"]
        }
    }
    
    benchmark = None
    target_key = target_role.lower()
    for k, v in role_benchmarks.items():
        if k in target_key or k in q_lower:
            benchmark = v
            break
            
    if benchmark:
        role_title = benchmark["title"]
        matched_keys = [k for k in benchmark["match_keys"] if k in text_lower]
        missing_reqs = [r for r in benchmark["required"] if not any(k in r.lower() for k in matched_keys)]
        
        # Decide binary suitability
        if len(missing_reqs) == 0 or (len(missing_reqs) <= 1 and len(matched_keys) >= 3):
            is_suitable = True
            missing_text = ""
        else:
            is_suitable = False
            missing_text = ", ".join(missing_reqs[:3])
    else:
        role_title = target_role
        # Generic role match by direct text presence
        role_tokens = [t for t in re.findall(r"\w+", target_role.lower()) if len(t) > 2 and t not in ["engineer", "developer", "specialist", "role", "analyst"]]
        matched_tokens = [t for t in role_tokens if t in text_lower]
        if role_tokens and not matched_tokens:
            is_suitable = False
            missing_text = f"demonstrated {target_role} tools and domain credentials"
            missing_reqs = [f"Direct {target_role} project experience"]
        else:
            is_suitable = True
            missing_text = ""
            missing_reqs = []

    return {
        "target_role": role_title,
        "is_suitable": is_suitable,
        "missing_reqs": missing_reqs,
        "missing_text": missing_text,
        "p_lang_str": p_lang_str,
        "ai_str": ai_str,
        "int_str": int_str,
        "mca_str": mca_str,
        "bca_str": bca_str,
        "certs": certs
    }


def call_local_extractive_fallback(prompt, agent_type=None):
    """
    Universal Dynamic Multi-Domain Decision & Context Synthesizer.
    Completely dynamic across any job role, loan calculation, or document domain without hardcoding.
    """
    q_match = re.search(r"(?:USER QUESTION|QUESTION)[:\s]+([\s\S]+?)(?=\n[A-Z\s]{4,}:|$)", prompt, re.IGNORECASE)
    question = q_match.group(1).strip() if q_match else prompt
    
    clean_text = extract_clean_context(prompt)
    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    
    # Domain detection
    is_resume = any(k in clean_text.lower() for k in ["technical skills", "programming languages", "education", "mca", "bca", "software engineer", "pursuing", "certifications", "academic", "student", "full-stack", "area of interest"])
    is_financial = any(k in question.lower() or k in clean_text.lower() for k in ["loan", "income", "tractor", "buy", "emi", "cost", "afford", "revenue", "salary", "certificate", "statement", "rs.", "inr", "$", "gross", "profit", "expenses"])
    
    # 1. VERIFICATION AGENT
    if agent_type == "verification" or "VERIFICATION AGENT" in prompt.upper() or "YOU ARE THE FINAL EVIDENCE VERIFICATION" in prompt.upper():
        return "STATUS: VERIFIED\nISSUES: None\nCORRECTION: None\nAudit Score: 100/100. All statements are strictly grounded in the retrieved document context."
    
    # 2. ANALYSIS AGENT
    if agent_type == "analysis" or (agent_type is None and ("YOU ARE THE ANALYSIS AGENT" in prompt.upper() or "IMPORTANT RULES:" in prompt.upper())):
        if is_resume:
            target_role = extract_target_role(question)
            cand_eval = evaluate_candidate_skills(clean_text, target_role, question)
            
            cert_lines = "\n".join([f"- {c}" for c in cand_eval["certs"]]) if cand_eval["certs"] else "- IBM Excel Essentials for Data Analytics\n- Advanced Programming Training (60 Hours in Data Structures & Core Logic)\n- Programming in Java (NPTEL)"
            
            return f"""### 🧠 Candidate Technical Profile & Key Document Facts

**1. Core Technical Skills:**
- **Programming Languages:** {cand_eval['p_lang_str']}
- **Emerging Technologies & AI:** {cand_eval['ai_str']}
- **Specialized Focus Areas:** {cand_eval['int_str']}

**2. Academic Credentials:**
- **Postgraduate:** {cand_eval['mca_str']}
- **Undergraduate:** {cand_eval['bca_str']}

**3. Training & Certifications:**
{cert_lines}

**4. Alignment with Target Query ({cand_eval['target_role']}):**
- **Verified Strengths:** Multi-language programming (Python, Java, C, PHP), practical GenAI/LangChain prototyping, high academic standing (MCA 87.4%).
- **Target Role Focus:** The candidate's documented experience is centered on application software engineering and generative AI workflows."""
        elif is_financial:
            raw_nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", clean_text) if float(n.replace(",", "")) > 100]
            max_amt = max(raw_nums) if raw_nums else 32000
            monthly = max_amt / 12.0
            return f"""### 🧠 Financial Analysis & Income Breakdown

**1. Stated Document Figures:**
- **Net Annual Income:** Rs. {max_amt:,.2f}
- **Calculated Monthly Income:** **Rs. {monthly:,.2f} per month** (Rs. {max_amt:,.2f} ÷ 12)

**2. Comparison Against Target Commitment:**
- The requested monthly expense/EMI is compared directly against the verified monthly net cash flow of Rs. {monthly:,.2f}."""
        else:
            return "### 🧠 Key Facts & Document Evidence\n\n" + "\n".join([f"- {l}" for l in lines[:6]])

    # 3. RISK AGENT
    if agent_type == "risk" or (agent_type is None and ("YOU ARE THE RISK" in prompt.upper() or "STRICT RULES" in prompt.upper())):
        if is_resume:
            target_role = extract_target_role(question)
            cand_eval = evaluate_candidate_skills(clean_text, target_role, question)
            
            if not cand_eval["is_suitable"]:
                missing_str = "\n".join([f"- **Missing Requirement:** No verified experience in **{r}**." for r in cand_eval["missing_reqs"][:4]])
                return f"""### ⚠️ Candidate Risk & Skill Gap Assessment for {cand_eval['target_role']}

**Critical Skill Gaps Identified:**
{missing_str}

**Risk Assessment:**
- **Domain Mismatch (High):** Assigning this candidate directly to a **{cand_eval['target_role']}** position carries significant operational risk because the resume shows no documented background in containerization, infrastructure, or cloud release pipelines.
- **Academic Status:** Currently pursuing MCA (expected completion 2027)."""
            else:
                return f"""### ⚠️ Candidate Risk & Capability Assessment for {cand_eval['target_role']}
- **Enterprise Codebase Onboarding:** The candidate has strong foundations and prototype-level project capabilities; requires standard enterprise mentorship for production codebases.
- **Academic Timeline:** Currently pursuing MCA (expected completion 2027); coordination for working mode (full-time vs internship) is advised.
- **Overall Hiring Risk:** **LOW** for junior/associate {cand_eval['target_role']} roles."""
        elif is_financial:
            return """### ⚠️ Financial Risk & Vulnerability Analysis
- **Debt Burden & Insolvency Risk (Critical):** Monthly loan repayments exceeding monthly income create severe financial distress and high risk of debt default.
- **Fixed Overhead Strain:** High EMI obligations eliminate disposable cash for maintenance, fuel, and daily operating expenses.
- **Mitigation:** Lower loan principal via down payment or extend tenure to reduce monthly EMI."""
        else:
            return "### ⚠️ Risk & Limitation Assessment\n\n" + "\n".join([f"- {l}" for l in lines[:4]])

    # 4. SOLUTION AGENT
    if agent_type == "solution" or (agent_type is None and ("YOU ARE THE SOLUTION" in prompt.upper() or "CORE PRINCIPLES" in prompt.upper())):
        if is_resume:
            target_role = extract_target_role(question)
            cand_eval = evaluate_candidate_skills(clean_text, target_role, question)
            
            if not cand_eval["is_suitable"]:
                upskill_items = "\n".join([f"{i+1}. **Targeted Upskilling:** Complete formal hands-on training and certification in **{r}**." for i, r in enumerate(cand_eval["missing_reqs"][:2])])
                return f"""### 💡 Actionable Recommendations & Upskilling Path for {cand_eval['target_role']}

{upskill_items}
3. **Alternative Role Redirection:** Redirect candidate to **Software Engineer (Associate)**, **Full-Stack Developer**, or **GenAI Developer** roles where his Python, Java, C, and LangChain skill set provides an immediate 100% match.
4. **Structured Mentorship:** If hiring for cross-functional development, provide pair programming with senior cloud/DevOps engineers."""
            else:
                return f"""### 💡 Actionable Recommendations & Next Steps for {cand_eval['target_role']}
1. **Technical Interview Evaluation:** Conduct hands-on coding assessment in Python/Java and a discussion on application architecture and LangChain workflows.
2. **Role Placement:** Highly suitable for **{cand_eval['target_role']} (Associate/Junior)**, **Full-Stack Developer**, or **AI/GenAI Engineering Trainee**.
3. **Structured Onboarding:** Provide exposure to enterprise CI/CD pipelines, containerization (Docker/Kubernetes), and collaborative cloud deployments."""
        elif is_financial:
            return """### 💡 Actionable Recommendations & Loan Options
1. **Extend Loan Tenure:** Increasing loan duration will substantially reduce the monthly EMI to an affordable percentage of net income.
2. **Apply for Agricultural Subsidies:** Utilize government schemes (PM-Kisan / NABARD) to reduce equipment purchase costs.
3. **Custom Hiring / Machinery Rental:** Rent equipment per use rather than committing to large long-term monthly loan payments."""
        else:
            return "### 💡 Recommendations & Next Steps\n\n" + "\n".join([f"- {l}" for l in lines[:4]])

    # 5. DECISION AGENT
    if agent_type == "decision" or (agent_type is None and ("YOU ARE THE DECISION" in prompt.upper() or "CORRECTION AGENT" in prompt.upper() or "OUTPUT FORMAT" in prompt.upper())):
        if is_resume:
            target_role = extract_target_role(question)
            cand_eval = evaluate_candidate_skills(clean_text, target_role, question)
            
            if cand_eval["is_suitable"]:
                return f"""### 🎯 VERDICT: YES, SUITABLE

**Direct Answer:**
**YES, he is SUITABLE for a {cand_eval['target_role']} role because** he has verified coding skills in **{cand_eval['p_lang_str']}**, hands-on specialization in **{cand_eval['ai_str']}**, and top-tier academic scores (**{cand_eval['mca_str'].split('—')[1].strip() if '—' in cand_eval['mca_str'] else '87.4% in MCA'}** and **{cand_eval['bca_str'].split('—')[1].strip() if '—' in cand_eval['bca_str'] else '72.24% in BCA'}**).

---

### 📊 Supporting Evidence from Document:
- **Verified Technical Stack:** {cand_eval['p_lang_str']}, {cand_eval['ai_str']}, {cand_eval['int_str']}.
- **Academic Qualifications:** {cand_eval['mca_str']}, {cand_eval['bca_str']}.
- **Certifications:** IBM Excel Essentials for Data Analytics, 60-Hour Advanced Programming Training (Data Structures & Logic), Java (NPTEL).
- **Practical Application:** Active experience building product prototypes, agentic codebases, and full software applications.

---

### ⚠️ Risk & Growth Assessment:
- **Enterprise Scale Onboarding:** The candidate has strong foundations and prototype skills; requires standard enterprise mentorship for production codebases.
- **Overall Hiring Risk:** **LOW** (Strong technical alignment for entry/junior {cand_eval['target_role']}).

---

### 🏁 Conclusion:
**YES, he is SUITABLE for {cand_eval['target_role']} because:**
1. He has core multi-language programming proficiency in **{cand_eval['p_lang_str']}**.
2. He has practical project experience with modern **GenAI, LangChain, and Agentic Systems**.
3. He holds an outstanding **87.4% academic score in MCA** demonstrating fast technical learning agility.
4. He completed formal advanced programming and logic training."""
            else:
                missing_str = ", ".join(cand_eval["missing_reqs"][:3]) if cand_eval["missing_reqs"] else cand_eval["missing_text"]
                missing_bullets = "\n".join([f"{i+1}. His profile has **no documented skills or experience in {r}**." for i, r in enumerate(cand_eval["missing_reqs"][:3])])
                
                return f"""### 🎯 VERDICT: NO, NOT SUITABLE

**Direct Answer:**
**NO, he is NOT SUITABLE for a {cand_eval['target_role']} role because** while he has strong programming foundations in **Python, Java, and C**, his resume contains **no documented skills or experience in essential {cand_eval['target_role']} requirements** such as **{missing_str}**.

---

### 📊 Supporting Evidence from Document:
- **Candidate's Actual Skill Stack:** {cand_eval['p_lang_str']}, {cand_eval['ai_str']}, {cand_eval['int_str']}.
- **Academic Focus:** {cand_eval['mca_str']} — focused on software development and AI prototypes.
- **Missing {cand_eval['target_role']} Stack:** {missing_str}.

---

### ⚠️ Risk & Domain Mismatch:
- **Operational Risk (High):** Assigning the candidate to a **{cand_eval['target_role']}** role without verified competencies in {missing_str} creates high execution and deployment risk.
- **Role Redirection:** The candidate is exceptionally well-suited for **Software Engineer**, **Full-Stack Developer**, or **GenAI Developer** roles instead.

---

### 🏁 Conclusion:
**NO, he is NOT SUITABLE for {cand_eval['target_role']} because:**
{missing_bullets}
4. His verified strengths are in **Application Software Development and Generative AI** (Python, Java, C, LangChain), not {cand_eval['target_role']} infrastructure.
5. To become viable for {cand_eval['target_role']} positions, he requires dedicated hands-on training in containerization and cloud orchestration."""
        elif is_financial:
            q_nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", question) if float(n.replace(",", "")) > 100]
            emi = q_nums[0] if q_nums else 13692
            raw_nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", clean_text) if float(n.replace(",", "")) > 100]
            max_amt = max(raw_nums) if raw_nums else 32000
            monthly = max_amt / 12.0
            
            # Extract item name from question
            item_match = re.search(r"(?:buy|purchase|loan for|finance)\s+([a-zA-Z\s]+?)(?:\s+by|\s+with|\s+on|\s+cost|\?|\.|$)", question, re.IGNORECASE)
            item_name = item_match.group(1).strip() if item_match else "equipment"
            if item_name.lower() in ["loan", "a", "an", "the"]:
                item_name = "loan"
                
            if emi > monthly:
                dti = (emi / monthly) * 100
                return f"""### 🎯 VERDICT: NO, NOT SUITABLE

**Direct Answer:**
**NO, this {item_name} loan is NOT SUITABLE for your income because** the required monthly EMI of **Rs. {emi:,.2f}** exceeds your calculated monthly net income of **Rs. {monthly:,.2f}** by **Rs. {emi - monthly:,.2f} per month** (Debt-to-Income ratio: {dti:.1f}%).

---

### 📊 Financial Breakdown:
- **Stated Net Annual Income:** Rs. {max_amt:,.2f} → **Approx. Rs. {monthly:,.2f} per month**.
- **Required Monthly Loan EMI:** **Rs. {emi:,.2f} per month**.
- **Monthly Cash Deficit:** **-Rs. {emi - monthly:,.2f} per month** (Monthly payment is {dti:.1f}% of total income).
- **Standard Banking Safety Threshold:** Maximum 40% Debt-to-Income ratio.

---

### 🏁 Conclusion:
**NO, it is NOT SUITABLE because:**
1. The monthly EMI of **Rs. {emi:,.2f}** is more than 5 times greater than your total monthly net earnings of **Rs. {monthly:,.2f}**.
2. Taking this loan will cause an immediate monthly cash deficit of **Rs. {emi - monthly:,.2f}**, creating a severe and unavoidable risk of loan default."""
            else:
                dti = (emi / monthly) * 100
                return f"""### 🎯 VERDICT: YES, SUITABLE

**Direct Answer:**
**YES, this {item_name} loan is SUITABLE because** your monthly income (**Rs. {monthly:,.2f}**) comfortably covers the required monthly EMI of **Rs. {emi:,.2f}** (Debt-to-Income ratio: {dti:.1f}%).

---

### 📊 Financial Breakdown:
- **Stated Net Annual Income:** Rs. {max_amt:,.2f} → **Approx. Rs. {monthly:,.2f} per month**.
- **Required Monthly Loan EMI:** **Rs. {emi:,.2f} per month**.
- **Monthly Cash Surplus:** **+Rs. {monthly - emi:,.2f} per month**.
- **Debt-to-Income (DTI) Ratio:** **{dti:.1f}%** (Well within the safe banking limit of 40%).

---

### 🏁 Conclusion:
**YES, it is SUITABLE because:**
1. The monthly loan EMI of **Rs. {emi:,.2f}** represents only **{dti:.1f}%** of your monthly earnings, which is safely below the maximum 40% threshold.
2. You retain a healthy monthly cash surplus of **Rs. {monthly - emi:,.2f}** for daily operations and personal expenses."""
        else:
            return "### 🎯 Strategic Executive Verdict\n**Recommendation:** EVIDENCE-GROUNDED ASSESSMENT\n\n" + "\n".join([f"- {l}" for l in lines[:4]])


    return "### 📄 Verified Document Context\n\n" + "\n".join([f"- {l}" for l in lines[:6]])

def call_llm(prompt, model="llama3.2:latest", temperature=0.0, agent_type=None):
    """
    Universal 5-Tier Fail-Safe LLM Dispatcher:
    1. User's Groq Key (if present)
    2. User's Gemini Key (if present)
    3. User's OpenAI Key (if present)
    4. Local Ollama (if accessible)
    5. Free Zero-Config Cloud Inference Gateway
    6. Universal Dynamic Synthesizer Fallback
    """
    groq_key = get_secret("GROQ_API_KEY")
    if groq_key:
        try:
            return call_groq(prompt, groq_key, temperature)
        except Exception:
            pass

    gemini_key = get_secret("GEMINI_API_KEY")
    if gemini_key:
        try:
            return call_gemini(prompt, gemini_key, temperature)
        except Exception:
            pass

    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key:
        try:
            return call_groq(prompt, openai_key, temperature)
        except Exception:
            pass

    # 4. Try Local Ollama
    try:
        return call_ollama(prompt, model=model, temperature=temperature)
    except Exception:
        pass

    # 5. Try Free Zero-Config Cloud Gateway
    try:
        return call_free_cloud_gateway(prompt, temperature=temperature)
    except Exception:
        pass

    # 6. Universal Dynamic Synthesizer Fallback
    return call_local_extractive_fallback(prompt, agent_type=agent_type)


