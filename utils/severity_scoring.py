# utils/severity_scoring.py

class SeverityCalculator:
    def __init__(self):
        pass

    def calculate_score(self, area_affected_pct, population_density, infra_density):
        # Basic algorithm to calculate a risk score (0-100)
        base_score = (area_affected_pct * 0.4) + (population_density / 100 * 0.4) + (infra_density / 10 * 0.2)
        
        # Cap the score at 100
        final_score = min(round(base_score, 2), 100.0)
        
        level = "LOW"
        if final_score > 75:
            level = "CRITICAL"
        elif final_score > 40:
            level = "MODERATE"
            
        return {
            "score": final_score,
            "risk_level": level
        }