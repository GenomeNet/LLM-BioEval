"""
Configuration for research projects displayed across the website.
This centralizes all research project metadata to avoid duplication.
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ResearchProject:
    """Represents a single research project."""
    id: str  # Unique identifier for the project
    title: str
    subtitle: str
    description: str
    authors: str
    doi: Optional[str]
    status: str  # 'published', 'active', 'development'
    status_label: str  # Display text for status
    route: Optional[str]  # Flask route (None if coming soon)
    color_theme: str  # 'purple', 'green', 'yellow'
    animation_type: str  # 'bacteria', 'dna', 'growth'
    link_text: str  # Text for the link button
    date_text: str  # "Updated regularly", "In development", etc.
    order: int  # Display order
    
    @property
    def card_type(self) -> str:
        """Get the card type label based on the project."""
        type_map = {
            'knowledge_calibration': 'Knowledge analysis',
            'phenotype_analysis': 'Phenotype Analysis',
            'growth_conditions': 'Phenotype analysis'
        }
        return type_map.get(self.id, 'Research')
    
    @property
    def status_class(self) -> str:
        """Get the CSS class for status indicator."""
        status_map = {
            'published': 'blue',
            'active': 'green',
            'development': 'yellow'
        }
        return status_map.get(self.status, 'blue')
    
    @property
    def is_available(self) -> bool:
        """Check if the project page is available."""
        return self.route is not None


# Define all research projects
RESEARCH_PROJECTS = [
    ResearchProject(
        id='phenotype_analysis',
        title='Evaluating LLM performance on fundamental microbial phenotype prediction',
        subtitle='Comprehensive benchmarking of language models predicting microbial characteristics',
        description='Comprehensive analysis of how language models predict broad microbial characteristics, from gram staining to pathogenicity, across thousands of species.',
        authors='P. C. Münch, N. Safaei, R. Mreches, M. Binder, Y. Han, G. Robertson, E. A. Franzosa, C. Huttenhower, A. C. McHardy',
        doi=None,  # Coming soon
        status='active',
        status_label='Phenotype Analysis',
        route='/phenotype_analysis',
        color_theme='green',
        animation_type='bacteria',
        link_text='Explore Analysis',
        date_text='Updated regularly',
        order=1
    ),
    ResearchProject(
        id='knowledge_calibration',
        title='Assessing LLM Knowledge Calibration for Microbial Taxonomy',
        subtitle='A dual test set of synthetic species and verified bacteria reveals when the model invents and when it defers.',
        description='Evaluating how much LLMs claim to know about bacteria by comparing their responses to internet data, revealing how frequently they generate unfounded claims about unknown species.',
        authors='P. C. Münch, N. Safaei, R. Mreches, M. Binder, Y. Han, G. Robertson, E. A. Franzosa, C. Huttenhower, A. C. McHardy',
        doi=None,  # Coming soon
        status='published',
        status_label='Knowledge analysis',
        route='/knowledge_calibration',
        color_theme='purple',
        animation_type='dna',
        link_text='View Results',
        date_text='Updated regularly',
        order=2
    ),
    ResearchProject(
        id='growth_conditions',
        title='Predicting bacterial growth conditions and metabolic flexibility',
        subtitle='Testing language models on environmental factors and metabolic pathways',
        description='Upcoming evaluation framework for testing LLM understanding of environmental factors, nutrient requirements, and metabolic pathways in bacteria.',
        authors='P. C. Münch, N. Safaei, R. Mreches, M. Binder, Y. Han, G. Robertson, E. A. Franzosa, C. Huttenhower, A. C. McHardy',
        doi=None,  # Coming soon (preprint)
        status='development',
        status_label='Coming Soon',
        route=None,  # Not available yet
        color_theme='yellow',
        animation_type='growth',
        link_text='Coming Soon',
        date_text='In development',
        order=3
    )
]

# Create lookup dictionary for quick access
RESEARCH_PROJECTS_BY_ID = {project.id: project for project in RESEARCH_PROJECTS}

# Create list of active projects (those with routes)
ACTIVE_PROJECTS = [p for p in RESEARCH_PROJECTS if p.is_available]

# Create list for homepage (show first 3 projects)
HOMEPAGE_PROJECTS = sorted(RESEARCH_PROJECTS, key=lambda x: x.order)[:3]

def get_project_by_id(project_id: str) -> Optional[ResearchProject]:
    """Get a research project by its ID."""
    return RESEARCH_PROJECTS_BY_ID.get(project_id)

def get_project_by_route(route: str) -> Optional[ResearchProject]:
    """Get a research project by its Flask route."""
    for project in RESEARCH_PROJECTS:
        if project.route == route:
            return project
    return None

def get_projects_for_page(page: str) -> List[ResearchProject]:
    """Get the appropriate list of projects for a specific page."""
    if page == 'index':
        return HOMEPAGE_PROJECTS
    elif page == 'research':
        return sorted(RESEARCH_PROJECTS, key=lambda x: x.order)
    else:
        return RESEARCH_PROJECTS