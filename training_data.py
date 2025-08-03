import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TrainingDataManager:
    def __init__(self, data_file="training_data.json"):
        self.data_file = data_file
        self.categories = {
            "health_insurance": [],
            "vehicle_mechanical": [],
            "constitution_law": [],
            "legal_cases": [],
            "policy_specific": []
        }
        self.load_data()
    
    def load_data(self):
        """Load existing training data if available"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.categories = data
                logger.info(f"Loaded {sum(len(questions) for questions in self.categories.values())} training questions")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing training data found or invalid format. Creating new dataset.")
            self.save_data()
    
    def save_data(self):
        """Save training data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.categories, f, indent=2)
        logger.info(f"Saved {sum(len(questions) for questions in self.categories.values())} training questions")
    
    def add_questions(self, category: str, questions: List[str]):
        """Add questions to a specific category"""
        if category not in self.categories:
            logger.warning(f"Category {category} not found. Creating new category.")
            self.categories[category] = []
        
        # Add only unique questions
        existing_questions = set(self.categories[category])
        added_count = 0
        
        for question in questions:
            if question not in existing_questions:
                self.categories[category].append(question)
                existing_questions.add(question)
                added_count += 1
        
        logger.info(f"Added {added_count} new questions to {category}")
        self.save_data()
        return added_count
    
    def get_questions(self, category: str = None, limit: int = None) -> List[str]:
        """Get questions from a specific category or all categories"""
        if category:
            if category not in self.categories:
                logger.warning(f"Category {category} not found")
                return []
            questions = self.categories[category]
        else:
            # Flatten all categories
            questions = []
            for cat_questions in self.categories.values():
                questions.extend(cat_questions)
        
        if limit and limit > 0:
            return questions[:limit]
        return questions
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the training data"""
        stats = {}
        for category, questions in self.categories.items():
            stats[category] = len(questions)
        stats["total"] = sum(stats.values())
        return stats

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize training data manager
    manager = TrainingDataManager()
    
    # Add health/insurance questions
    health_questions = [
        "When will my root canal claim of Rs 25,000 be settled?",
        "I have done an IVF for Rs 56,000. Is it covered?",
        "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
        "Give me a list of documents to be uploaded for hospitalization for heart surgery."
    ]
    manager.add_questions("health_insurance", health_questions)
    
    # Add vehicle/mechanical questions
    vehicle_questions = [
        "The ideal spark plug gap recommended is 0.8-0.9 mm.",
        "Does this come in a tubeless tyre version?",
        "Is it compulsory to have a disc brake?",
        "Can I put Thums Up instead of oil?"
    ]
    manager.add_questions("vehicle_mechanical", vehicle_questions)
    
    # Add constitution & law questions
    constitution_questions = [
        "What is the official name of India according to Article 1 of the Constitution?",
        "Which Article guarantees equality before the law and equal protection of laws to all persons?",
        "What is abolished by Article 17 of the Constitution?",
        "What are the key ideals mentioned in the Preamble of the Constitution of India?",
        "According to Article 24, children below what age are prohibited from working in hazardous industries?",
        "What is the significance of Article 21 in the Indian Constitution?",
        "Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for under this Article?",
        "Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship?",
        "What restrictions can the State impose on the right to freedom of speech under Article 19(2)?"
    ]
    manager.add_questions("constitution_law", constitution_questions)
    
    # Add legal case questions
    legal_questions = [
        "If my car is stolen, what case will it be in law?",
        "If I am arrested without a warrant, is that legal?",
        "If someone denies me a job because of my caste, is that allowed?",
        "If the government takes my land for a project, can I stop it?",
        "If my child is forced to work in a factory, is that legal?",
        "If I am stopped from speaking at a protest, is that against my rights?",
        "If a religious place stops me from entering because I'm a woman, is that constitutional?",
        "If I change my religion, can the government stop me?",
        "If the police torture someone in custody, what right is being violated?",
        "If I'm denied admission to a public university because I'm from a backward community, can I do something?"
    ]
    manager.add_questions("legal_cases", legal_questions)
    
    # Add policy specific questions
    policy_questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    manager.add_questions("policy_specific", policy_questions)
    
    # Print statistics
    print("Training data statistics:")
    for category, count in manager.get_stats().items():
        print(f"{category}: {count} questions")