"""
VidyāSetu Algorithmic Core Engine
Contains mathematically rigorous implementations of:
1. Bayesian Knowledge Tracing (BKT)
2. Item Response Theory (2PL IRT) & Computerized Adaptive Testing (CAT)
3. Directed Acyclic Graph (DAG) Prerequisite Traversal
4. Teacher 3-Cohort Clustering Engine
"""

import math
from typing import List, Dict, Any, Tuple, Optional

class BayesianKnowledgeTracer:
    """
    Standard Corbett & Anderson Bayesian Knowledge Tracing model.
    Parameters:
      p_init (L0): Probability student initially knows the concept.
      p_transit (T): Probability student transitions from unlearned to learned.
      p_slip (S): Probability student makes a mistake despite knowing concept.
      p_guess (G): Probability student guesses correctly without knowing concept.
    """
    def __init__(self, p_init: float = 0.25, p_transit: float = 0.15, p_slip: float = 0.10, p_guess: float = 0.20):
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update(self, p_learned: float, is_correct: bool) -> float:
        """
        Updates P(L_t) given observation at time t.
        Formula:
          If correct:
            P(L_t | obs=1) = (P(L_{t-1}) * (1 - S)) / (P(L_{t-1}) * (1 - S) + (1 - P(L_{t-1})) * G)
          If incorrect:
            P(L_t | obs=0) = (P(L_{t-1}) * S) / (P(L_{t-1}) * S + (1 - P(L_{t-1})) * (1 - G))
          Then apply transition:
            P(L_t) = P(L_t | obs) + (1 - P(L_t | obs)) * T
        """
        if is_correct:
            numerator = p_learned * (1.0 - self.p_slip)
            denominator = numerator + (1.0 - p_learned) * self.p_guess
        else:
            numerator = p_learned * self.p_slip
            denominator = numerator + (1.0 - p_learned) * (1.0 - self.p_guess)

        if denominator == 0:
            p_posterior = p_learned
        else:
            p_posterior = numerator / denominator

        # Account for learning during the step
        p_updated = p_posterior + (1.0 - p_posterior) * self.p_transit
        return min(max(p_updated, 0.001), 0.999)

    def predict_correct(self, p_learned: float) -> float:
        """Calculates expected probability that the student will answer correctly next."""
        return p_learned * (1.0 - self.p_slip) + (1.0 - p_learned) * self.p_guess


class ItemResponseTheoryCAT:
    """
    2-Parameter Logistic (2PL) Item Response Theory model.
    theta: Latent student ability.
    b: Item difficulty.
    a: Item discrimination.
    """
    @staticmethod
    def probability_correct(theta: float, b: float, a: float = 1.0) -> float:
        """P(Y=1 | theta, b, a) = 1 / (1 + exp(-a * (theta - b)))"""
        exponent = -a * (theta - b)
        # Numerical stability clamp
        exponent = max(min(exponent, 35.0), -35.0)
        return 1.0 / (1.0 + math.exp(exponent))

    @staticmethod
    def fisher_information(theta: float, b: float, a: float = 1.0) -> float:
        """I(theta) = a^2 * P(theta) * (1 - P(theta))"""
        p = ItemResponseTheoryCAT.probability_correct(theta, b, a)
        return (a ** 2) * p * (1.0 - p)

    @staticmethod
    def update_theta(theta: float, is_correct: bool, b: float, a: float = 1.0, lr: float = 0.5) -> float:
        """Maximum Likelihood / gradient update for student ability theta."""
        p = ItemResponseTheoryCAT.probability_correct(theta, b, a)
        error = (1.0 if is_correct else 0.0) - p
        new_theta = theta + lr * (a * error)
        # Clamped between -3.0 and +3.0 (standard standard normal scale)
        return max(min(new_theta, 3.0), -3.0)


class PrerequisiteDAG:
    """
    Represents the curriculum dependency graph across Classes 6 to 10.
    Identifies root-cause foundational gaps when a student struggles with higher concepts.
    """
    def __init__(self):
        # topic_id -> dict with details and parent prerequisite_ids
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def add_topic(self, topic_id: str, name: str, grade: int, subject: str, prerequisites: List[str]):
        self.nodes[topic_id] = {
            "id": topic_id,
            "name": name,
            "grade": grade,
            "subject": subject,
            "prerequisites": prerequisites
        }

    def trace_root_gap(self, failed_topic_id: str, student_mastery_map: Dict[str, float], threshold: float = 0.65) -> Optional[Dict[str, Any]]:
        """
        Traverses backwards through prerequisites (DFS) to find the earliest/lowest-grade
        unmastered prerequisite causing the current failure.
        """
        if failed_topic_id not in self.nodes:
            return None

        visited = set()
        queue = list(self.nodes[failed_topic_id]["prerequisites"])
        root_gap = None
        lowest_grade = 999

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited or curr_id not in self.nodes:
                continue
            visited.add(curr_id)

            curr_node = self.nodes[curr_id]
            mastery = student_mastery_map.get(curr_id, 0.0)

            # If this prerequisite is below mastery threshold
            if mastery < threshold:
                if curr_node["grade"] < lowest_grade:
                    lowest_grade = curr_node["grade"]
                    root_gap = {
                        "topic": curr_node,
                        "mastery": mastery,
                        "gap_grade": curr_node["grade"]
                    }
                # Check deeper prerequisites of this failed node
                queue.extend(curr_node["prerequisites"])

        return root_gap


class TeacherClusteringEngine:
    """
    Groups a classroom of 40-60 students into 3 actionable cohorts:
    1. Cohort Sprint (Mastered, ready for advanced/Olympiad work)
    2. Cohort Guided (On-track with standard textbook syllabus)
    3. Cohort Bridge (Common root gap requiring foundational intervention)
    """
    @staticmethod
    def cluster_classroom(students_data: List[Dict[str, Any]], current_topic_id: str, dag: PrerequisiteDAG) -> Dict[str, Any]:
        sprint = []
        guided = []
        bridge = []

        gap_counts: Dict[str, int] = {}

        for student in students_data:
            mastery = student.get("mastery_scores", {}).get(current_topic_id, 0.5)
            confidence = student.get("confidence", 0.5)

            # High mastery and high confidence -> Sprint
            if mastery >= 0.80 and confidence >= 0.7:
                sprint.append(student)
            elif mastery >= 0.60:
                guided.append(student)
            else:
                # Student is struggling; trace root gap
                root_gap = dag.trace_root_gap(current_topic_id, student.get("mastery_scores", {}))
                gap_name = root_gap["topic"]["name"] if root_gap else "Core Practice"
                gap_counts[gap_name] = gap_counts.get(gap_name, 0) + 1
                student_copy = dict(student)
                student_copy["diagnosed_gap"] = gap_name
                student_copy["gap_grade"] = root_gap["topic"]["grade"] if root_gap else student["grade"]
                bridge.append(student_copy)

        # Identify most common shared gap in Bridge cohort
        top_shared_gap = max(gap_counts.items(), key=lambda x: x[1])[0] if gap_counts else "Fundamental Review"

        # Generate Teacher 3-Minute Blackboard Script
        script = TeacherClusteringEngine._generate_whiteboard_script(current_topic_id, top_shared_gap, len(bridge))

        return {
            "total_students": len(students_data),
            "sprint": {
                "count": len(sprint),
                "percentage": round(len(sprint) / len(students_data) * 100, 1) if students_data else 0,
                "students": sprint,
                "action": "Assign Level-3 Higher Order Thinking (HOTS) / Olympiad challenge worksheet."
            },
            "guided": {
                "count": len(guided),
                "percentage": round(len(guided) / len(students_data) * 100, 1) if students_data else 0,
                "students": guided,
                "action": "Continue with prescribed NCERT textbook exercises and standard problem set."
            },
            "bridge": {
                "count": len(bridge),
                "percentage": round(len(bridge) / len(students_data) * 100, 1) if students_data else 0,
                "students": bridge,
                "primary_gap": top_shared_gap,
                "whiteboard_script": script
            }
        }

    @staticmethod
    def _generate_whiteboard_script(topic_id: str, gap_name: str, affected_count: int) -> Dict[str, Any]:
        return {
            "duration": "3 Minutes (Opening of Tomorrow's Class)",
            "goal": f"Clear shared misunderstanding on '{gap_name}' affecting {affected_count} students.",
            "board_diagram": (
                "Step 1: Draw horizontal number line: -3, -2, -1, 0, 1, 2, 3.\n"
                "Step 2: Contrast (+3) - (+5) = -2 (Walking left on debt line).\n"
                "Step 3: Connect directly to the equation: 3x - 12 = 15 -> moving -12 to RHS becomes +12."
            ),
            "teacher_talk_track": (
                "Good morning class! Notice how signs behave like financial debts: "
                "borrowing rupees means negative. When moving a balance across the '=' bridge, "
                "the operation reverses. Let's do this 30-second sign check before our next problem."
            ),
            "rapid_check_question": "Quick show of fingers: what is (-4) * (-3)? (A: +12, B: -12)"
        }


class VernacularDictionaryEngine:
    """
    U-Dictionary inspired instant vernacular translation & conceptual anchor engine.
    Bridges English STEM vocabulary to Indian languages with everyday analogies.
    """
    TERMS = {
        "variable": {
            "term": "Variable",
            "phonetic": "VAIR-ee-uh-bul",
            "hi": "चर (Char)",
            "te": "చలరాశి (Chalarasi)",
            "definition": "A quantity or symbol (like x or y) that can change or take different values.",
            "analogy": "Like an empty dabba (tiffin box) whose contents you have to find out."
        },
        "coefficient": {
            "term": "Coefficient",
            "phonetic": "koh-ih-FISH-unt",
            "hi": "गुणांक (Gunaank)",
            "te": "గుణకం (Gunakam)",
            "definition": "A numerical quantity placed before and multiplying the variable (e.g., 3 in 3x).",
            "analogy": "Like the number of packets of biscuits you bought (3 packets of x biscuits)."
        },
        "transposition": {
            "term": "Transposition",
            "phonetic": "trans-poh-ZISH-un",
            "hi": "पक्षांतरण (Pakshaantaran)",
            "te": "పక్షాంతరీకరణ (Pakshaantarikarana)",
            "definition": "Moving a term from one side of an equation to the other, which reverses its operation (+ to -, * to /).",
            "analogy": "Crossing an interstate toll bridge: subtraction becomes addition!"
        },
        "linear equation": {
            "term": "Linear Equation",
            "phonetic": "LIN-ee-er ih-KWAY-zhun",
            "hi": "रैखिक समीकरण (Raikhik Samikaran)",
            "te": "సరళ సమీకరణం (Sarala Sameekaranam)",
            "definition": "An algebraic equation in which the highest power of the variable is 1, creating a straight line on a graph.",
            "analogy": "Like a laser beam that never curves—moves in a single straight path."
        },
        "reciprocal": {
            "term": "Reciprocal",
            "phonetic": "rih-SIP-ruh-kul",
            "hi": "व्युत्क्रम (Vyutkram)",
            "te": "గుణకార విలోమం (Gunakara Vilomam)",
            "definition": "The inverse of a number such that their product is 1 (e.g., reciprocal of 2/3 is 3/2).",
            "analogy": "Doing a headstand! The fraction turns upside down."
        },
        "integers": {
            "term": "Integers",
            "phonetic": "IN-tih-jerz",
            "hi": "पूर्णांक (Poornaank)",
            "te": "పూర్ణ సంఖ్యలు (Poorna Sankhyalu)",
            "definition": "Whole numbers that can be positive, negative, or zero (..., -2, -1, 0, 1, 2, ...).",
            "analogy": "Positive numbers are profits; negative numbers are debts you owe."
        }
    }

    @classmethod
    def lookup(cls, word: str) -> Optional[Dict[str, Any]]:
        cleaned = word.lower().strip()
        for k, v in cls.TERMS.items():
            if k in cleaned or cleaned in k:
                return v
        return None

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        return list(cls.TERMS.values())


class SocraticDoubtSolver:
    """
    Byju's / Vedantu inspired Socratic Doubt Assistant.
    Guides the student with prompts and analogies instead of spoon-feeding direct answers.
    """
    KNOWLEDGE_BASE = [
        {
            "keywords": ["minus", "negative", "sign", "crossing", "equal"],
            "topic": "Sign Flipping across '='",
            "socratic_prompt": "Think about a balancing scale. If you remove a 12kg debt from the left side, what must you do to the right to keep it equal?",
            "analogy": "Imagine borrowing ₹12 from a friend. If the bank clears that loan, your net balance increases by ₹12!",
            "check_question": "If x - 5 = 10, does x = 10 - 5 or x = 10 + 5?"
        },
        {
            "keywords": ["bracket", "parenthesis", "expand", "multiply"],
            "topic": "Distributive Property in Equations",
            "socratic_prompt": "When you have 3(x - 4), does the 3 outside multiply only the x, or every person sitting inside the room?",
            "analogy": "Like a teacher distributing 3 samosas to EVERY student inside the classroom!",
            "check_question": "What is 3 multiplied by (-4)? Is it +12 or -12?"
        },
        {
            "keywords": ["fraction", "divide", "reciprocal", "two sides"],
            "topic": "Solving Fractional Equations",
            "socratic_prompt": "To get rid of a fraction like (x / 4) = 5, what operation is the opposite of division?",
            "analogy": "If you split a pizza among 4 people, to get the whole pizza back you multiply by 4!",
            "check_question": "If x/3 = 6, what is x?"
        }
    ]

    @classmethod
    def resolve_doubt(cls, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        for item in cls.KNOWLEDGE_BASE:
            if any(k in q_lower for k in item["keywords"]):
                return {
                    "matched": True,
                    "topic": item["topic"],
                    "socratic_nudge": item["socratic_prompt"],
                    "desi_analogy": item["analogy"],
                    "rapid_check": item["check_question"]
                }
        return {
            "matched": False,
            "topic": "General Equation Guidance",
            "socratic_nudge": "What is the single term with the variable 'x' you are trying to isolate on the left?",
            "desi_analogy": "Equations are like Rythu Bazaar weighing scales: whatever you do to one side, do to the other!",
            "rapid_check": "Try moving constant numbers to the right-hand side first."
        }


class GyanReelManager:
    """
    Instagram-style vertical micro-learning feed (15-30s bites).
    """
    REELS = [
        {
            "id": "reel_1",
            "title": "Why is (-1) × (-1) = +1? The Debt Hack",
            "creator": "Smt. Lakshmi (Math HOD)",
            "duration": "00:22",
            "likes": 1420,
            "badge": "Mind Blown 🤯",
            "analogy": "Cancelling a negative debt is a positive gain!",
            "quiz": "What is (-5) × (-4)?",
            "options": ["+20", "-20", "-9", "+9"],
            "correct": 0
        },
        {
            "id": "reel_2",
            "title": "Solve 3x - 12 = 15 in 15 Seconds (No Pen Needed)",
            "creator": "Aarav Sharma (Class 8 Peer Tutor)",
            "duration": "00:18",
            "likes": 980,
            "badge": "Board Exam Hack ⚡",
            "analogy": "Send 12 to right side: 15 + 12 = 27. Divide by 3 = 9. Boom!",
            "quiz": "What is x if 2x - 8 = 10?",
            "options": ["9", "1", "18", "8"],
            "correct": 0
        },
        {
            "id": "reel_3",
            "title": "How Google Maps Uses Class 9 Coordinate Geometry",
            "creator": "Canvas Science Lab",
            "duration": "00:28",
            "likes": 2340,
            "badge": "Real World 🌍",
            "analogy": "Latitude is your X-axis, Longitude is your Y-axis!",
            "quiz": "In (x, y), which axis tells you how far left or right you are?",
            "options": ["X-axis", "Y-axis", "Z-axis", "Origin"],
            "correct": 0
        }
    ]

    @classmethod
    def get_feed(cls) -> List[Dict[str, Any]]:
        return cls.REELS
