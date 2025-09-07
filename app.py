import os
from dotenv import load_dotenv
import gradio as gr # NEW: Import Gradio
from crewai import Agent, Task, Crew, Process
from langchain_litellm import ChatLiteLLM
import docx
import time, random
from typing import Callable


load_dotenv()

OVERLOAD_TOKENS = ("overloaded", "try again later", "rate limit", "quota", "429", "503", "unavailable")

def is_overload_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(tok in msg for tok in OVERLOAD_TOKENS)

def make_llm(model_name: str, api_key: str):
    
    return ChatLiteLLM(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
        max_tokens=4096
    )

KEYWORDS_POLICY = lambda kw: (
    "Must-use Keywords:\n"
    f"{kw if kw else '[None provided]'}\n\n"
    "Usage Rules:\n"
    "• Scatter these exact phrases and close variants across ALL sections: Summary, Skills, and every role’s bullets.\n"
    "• Prioritize natural usage aligned to truthful experience; do NOT fabricate employers, titles, or tools.\n"
    "• Prefer exact phrases where reasonable; otherwise use close variants/stems (e.g., 'Kubernetes'/'K8s').\n"
    "• Limit to 1–2 keyword insertions per bullet to avoid stuffing; prefer distribution over repetition.\n"
    "• If a keyword is only tangentially relevant, place it in Skills/Projects rather than forcing it into a bullet.\n"
    "• If a keyword is truly irrelevant to the candidate and JD, list it under 'Skills to Develop (Gap Analysis)'.\n"
)

def kickoff_with_retries(
    build_crew_fn: Callable[[], Crew],
    max_attempts: int = 6,
    base_delay: float = 1.8,
    jitter: float = 0.6,
    switch_model_after_attempts: int = 3
):
    """
    Attempts to kickoff() a Crew multiple times with exponential backoff.
    Rebuilds the crew with a fallback model after `switch_model_after_attempts`.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            crew = build_crew_fn()
            return crew.kickoff()
        except Exception as e:
            last_err = e
            if not is_overload_error(e):
                # Not an overload/rate issue → fail fast
                raise
            # overload/rate-limited → backoff and retry
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            print(f"[Attempt {attempt}/{max_attempts}] Provider overloaded. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    
    raise RuntimeError(f"Failed after {max_attempts} attempts due to provider overload: {last_err}")


# --- THE CORE AGENTIC LOGIC, WRAPPED IN A FUNCTION ---

def run_resume_crew(job_description, resume_file, keywords):
    """
    Takes a job description and a resume file path,
    runs the crewAI workflow, and returns the tailored LaTeX code.
    """

    keywords_clean = ", ".join([
        k.strip() for k in (keywords or "").splitlines() if k.strip()
    ])

    kw_rules = KEYWORDS_POLICY(keywords_clean)


    # --- 1. Load Data ---
    # Read the text from the uploaded .docx file
    try:
        doc = docx.Document(resume_file.name) # .name gets the temporary file path from Gradio
        all_paragraphs = [p.text for p in doc.paragraphs]
        resume_text = "\n".join(all_paragraphs)
    except Exception as e:
        return f"Error reading DOCX file: {e}. Please ensure it's a valid .docx file."

    # --- 2. Initialize LLM ---
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        return "ERROR: GOOGLE_API_KEY not found. Please check your .env file."
    
    primary_model = "gemini/gemini-1.5-flash-latest"
    fallback_model = "gemini/gemini-1.5-pro-latest"
    use_fallback = {"value": False}

    def build_crew():
        model_name = fallback_model if use_fallback["value"] else primary_model
        llm = make_llm(model_name, google_api_key)


        # --- 3. Defining Agents ---
        profile_analyst = Agent(
            role="Profile Analyst",
            goal="Extract a candidate’s full professional profile (summary, skills, projects, education, and work experience) from a resume in a structured, analysis-ready format.",
            backstory="You are a senior career coach and expert resume analyst. You can instantly see hidden strengths, categorize technical and soft skills, and highlight quantified achievements for career positioning.",
            llm=llm
        )
        job_analyst = Agent(
            role="Job Analyst",
            goal="Dissect job descriptions to uncover the hiring manager’s true priorities. Identify both explicit and implicit requirements, trending technical keywords, and key action verbs.",
            backstory="You are a top-tier technical recruiter. You know how to interpret job postings, spot the hidden must-haves, and prioritize the employer’s most critical needs.",
            llm=llm
        )
        resume_strategist = Agent(
            role="Resume Strategist",
            goal=f"Craft a resume content strategy tailored to the job description. Ensure 80 percent alignment with job description skills/keywords and 20 percent inclusion of the candidate’s unique skills. Scatter job keywords naturally across all sections. Add at least 9-10 points in the work experience section. Rewrite bullets to highlight quantifiable results, trending tools, and impact in recruiter-friendly language.Make sure to include these keywords in the resume as well.\n\n{kw_rules}",
            backstory="You are a master resume writer and strategist. You blend ATS optimization with a compelling human narrative. You ensure strong verbs, reasonable metrics, and trending tech keywords that resonate with recruiters and hiring managers.",
            llm=llm
        )
        # The ATS Judge Agent--------------refine
        ats_judge = Agent(
            role='ATS Judge',
            goal="Score a resume against a job description and provide actionable feedback for improvement to exceed a 90% ATS score.",
            backstory="You are a highly advanced ATS screening algorithm, 'ResumeRanker 9000'. You analyze resumes with cold, hard logic, focusing purely on keyword matching, skill alignment, and relevance of experience. Your feedback is direct, numerical, and aimed at maximizing the score.",
            llm=llm,
            verbose=True
        )
        latex_specialist = Agent(
            role="LaTeX Formatting Specialist",
            goal="Convert structured resume content into a polished, single-file LaTeX resume "
                "with clean formatting. Ensure sections are well-balanced and ATS-compatible.",
            backstory="You are a document design expert specializing in LaTeX resumes. You create modern, professional designs that recruiters love and ATS can parse cleanly.",
            llm=llm
        )

        # --- 4. Define Tasks ---
        profile_extraction_task = Task(
            description=f"Analyze the candidate’s master resume text. Extract and structure all key information into:- Professional Summary (rewrite in concise, recruiter-friendly tone).- Technical Skills (grouped by category).- Soft Skills.- Work Experience (detailed achievements with measurable impact).- Education & Certifications.Here is the resume:\n\n{resume_text}",
            expected_output="A clean, well-organized structured profile summary with clearly categorized sections.",
            agent=profile_analyst

        )
        job_analysis_task = Task(
            description=f"Analyze the job description to identify:- Top 15-20 technical skills (with trending industry keywords).- Top 3–5 soft skills.- Key phrases and action verbs from responsibilities.\n\nPrioritize from most to least important.\n\nHere is the job description:\n\n{job_description}\n\nAlso, adhere to these keyword usage rules:\n\n"
                        f"{kw_rules}",
            expected_output="A prioritized bullet-point list of required skills and responsibilities.",
            agent=job_analyst
        )
        drafting_task = Task(
            description="Create a detailed, tailored resume content draft. Crucially, perform a gap analysis. The 'Skills' section MUST contain three sub-sections: 'Matched Skills' (skills the candidate has that the job requires), 'Skills to Develop (Gap Analysis)' (skills the job requires that the candidate does not have), and 'Additional Skills' (other skills the candidate has). Rephrase work experience to align with the job's keywords. Make the Job description 80 percent in the resume.Include candidates unmatched skills as well in the experience section if the resume doesn't go over 2 pages. Your output is the first draft of the full resume content."
            "• Truthfulness: do not invent employers/titles; only generalize responsibly.\n\n"
            f"Take into account the following keywords and make sure, all the keywords are covered in the resume:\n\n{kw_rules}",
            expected_output="A complete, structured text document for the first draft of the resume, including the detailed three-part skills section.",
            agent=resume_strategist,
            context=[profile_extraction_task, job_analysis_task]
        )
        content_strategy_task = Task(
            description="Using the candidate profile and the job requirements, create tailored resume content:1. Professional Summary (2–3 sentences). Blend 80% JD keywords with 20 percent unique skills.2. Skills section .3. Work Experience – rewrite each bullet using strong verbs, quantified impact, and JD keywords. Ensure trending terms (e.g., cloud, AI/ML, microservices, DevOps, cybersecurity depending on context). 4. Education & Certifications – format clearly, include job-relevant credentials.Rules:- Scatter JD keywords naturally across all sections.- Maintain 80 percent alignment with JD, 20 percent candidate’s extra skills. - Use strong verbs (‘engineered’, ‘optimized’, ‘secured’, ‘delivered’).- Quantify with reasonable numbers (%, $, time saved, performance improved).- Keep tone professional, concise, and ATS-friendly.",
            expected_output="A complete, structured text document containing the new summary, prioritized skills list, rephrased experience bullets, and education/certifications.",
            agent=resume_strategist,
            context=[profile_extraction_task, job_analysis_task]
        )# This task depends on the first two
        # First Pass: Draft
        drafting_task = Task(
            description="Create a detailed, tailored resume content draft. Perform a gap analysis in the 'Skills' section with three sub-sections: 'Matched Skills', 'Skills to Develop (Gap Analysis)', and 'Additional Skills'. This is the first version.",
            expected_output="A complete, structured text document for the first draft of the resume.",
            agent=resume_strategist,
            context=[profile_extraction_task, job_analysis_task]
        )

        # --- First Refinement Loop ---
        judging_pass_1 = Task(
            description="Evaluate the FIRST draft against the ORIGINAL job description.\n"
                        "Return:\n"
                        "• ATS Score (0–100), target ≥95\n"
                        "• Coverage Gaps: missing high-priority JD keywords\n"
                        "• Must-use Keywords Report: for each provided keyword, mark Covered (Y/N), Section(s) used, and Suggested fix if missing\n"
                        "• Bullet Count Check: list any roles with <6 bullets\n"
                        "• Actionable Fixes: specific terms to add, sections to adjust, bullets to expand\n\n"
                        f"Original Job Description:\n{job_description}\n\n"
                        f"{kw_rules}"  ,
            expected_output="A numerical score and a bulleted list of actionable improvements if the score is below 90.",
            agent=ats_judge,
            context=[drafting_task]
        )
        revision_pass_1 = Task(
            description="Revise the first resume draft based on the feedback from the first judging pass. Implement all suggested improvements to create a second, improved version of the resume content."
                        f"{kw_rules}",
            expected_output="The second, revised, and improved resume content as a structured text document.",
            agent=resume_strategist,
            context=[drafting_task, judging_pass_1]
        )

        # --- Second Refinement Loop ---
        judging_pass_2 = Task(
            description="Re-evaluate the SECOND draft. Require ≥95 ATS and ≥6 bullets per role. "
                        "Recheck the must-use keywords coverage; all should be covered or responsibly placed in Gap Analysis with justification. "
                        "If not met, provide final critical fixes only.\n\n"
                        f"Original Job Description:\n{job_description}\n\n"
                        f"{kw_rules}",
            expected_output="A final numerical score and, if necessary, a final bulleted list of critical improvements.",
            agent=ats_judge,
            context=[revision_pass_1] # Depends on the first revision
        )
        final_polish_task = Task(
            description="Final polish: apply judging_pass_2 fixes. Ensure ATS ≥95, each role has 6–10 bullets, 80/20 JD vs candidate, "
        "crisp phrasing, quantified impact, and correct, non-fabricated usage of tools/tech.\n\n"
        f"{kw_rules}" ,
            expected_output="The final, polished, and highest-scoring resume content as a structured text document.",
            agent=resume_strategist,
            context=[revision_pass_1, judging_pass_2] # Depends on the first revision and second judgment
        )

        latex_generation_task = Task(
            description=f"Convert the tailored resume content into a complete, single-file LaTeX document using a standard, clean resume template. The final output must be ONLY the raw LaTeX code. Fix the percentage and special character issues as well. Format should be like this:\n\n{format_text}",
            expected_output="A single block of pure, raw LaTeX code that is ready for compilation.",
            agent=latex_specialist,
            context=[content_strategy_task]
        )

        # --- 5. Create and Run Crew ---
        resume_crew = Crew(
            agents=[profile_analyst, job_analyst, resume_strategist, ats_judge, latex_specialist],
            tasks=[
                profile_extraction_task,
                job_analysis_task,
                drafting_task,
                judging_pass_1,
                revision_pass_1,
                judging_pass_2,
                final_polish_task,
                content_strategy_task,
                latex_generation_task
            ],
            process=Process.sequential,
            verbose=True
        )
        return resume_crew
    
    attempts_before_switch = 3
    def build_crew_with_switch():
        # Flip to fallback if we’ve already tried `attempts_before_switch` times (handled in kickoff_with_retries)
        return build_crew()
    
    try:
        # result = resume_crew.kickoff()
        result = kickoff_with_retries(build_crew_with_switch, max_attempts=attempts_before_switch)
        latex_code = clean_latex_output(str(result))
        return latex_code

    except Exception as e:
        return f"An error occurred during crew execution: {e}"
    

def clean_latex_output(latex: str) -> str:
    # Remove triple backtick fences if present
    if latex.strip().startswith("```"):
        latex = latex.strip("`")  # remove backticks
        # Optional: if fenced with "latex", strip the language too
        latex = latex.replace("latex", "", 1).strip()
    return latex.strip()


# --- THE GRADIO WEB UI ---

# Defining the user interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI-Powered Resume Tailoring Agent")
    gr.Markdown("Upload your master resume, paste the target job description, and let the AI craft a tailored LaTeX resume for you.")
    
    with gr.Row():
        # Inputs
        with gr.Column():
            gr.Markdown("### Inputs")
            job_description_input = gr.Textbox(lines=15, label="Paste Job Description Here")
            keywords_input = gr.Textbox(lines=15, label="Paste Keywords Here")
            resume_file_input = gr.File(label="Upload Your Master Resume (.docx)", file_types=[".docx"])
            submit_button = gr.Button("✨ Tailor My Resume", variant="primary")
        
        # Outputs
        with gr.Column():
            gr.Markdown("### Outputs")
            latex_output = gr.Textbox(lines=20, label="Generated LaTeX Code", interactive=False)

        with open('format.txt', 'r', encoding='utf-8') as f:
            format_text = f.read()

    

    # Defining the button's click action
    submit_button.click(
        fn=run_resume_crew,
        inputs=[job_description_input, resume_file_input, keywords_input],
        outputs=latex_output
    )

# --- LAUNCH THE APP ---

if __name__ == "__main__":
    
    demo.launch()