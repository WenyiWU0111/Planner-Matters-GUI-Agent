"""Helper tools for the Qwen-Agent framework"""
from typing import Dict, Any
import json
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool


@register_tool('verifier')
class VerifierTool(BaseTool):
    """Tool for verifying actions and trajectories to get feedback when stuck"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'verifier'
        self.description = 'Verify the last action or recent trajectory to get feedback when stuck or at a loss. Use this to get suggestions on what went wrong and how to proceed.'
        self.parameters = {
            'type': 'object',
            'properties': {
                'verification_type': {
                    'type': 'string',
                    'enum': ['action', 'trajectory', 'both'],
                    'description': 'Type of verification: "action" to verify last action, "trajectory" to verify recent trajectory, "both" to verify both (default: "both")'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for the verification'
                }
            },
            'required': ['verification_type']
        }
        
    def call(self, args: str, **kwargs) -> str:
        """Acknowledge verifier action - actual verification happens in _process_response"""
        try:
            # Parse arguments for logging
            if isinstance(args, str):
                import json
                args = json.loads(args)
            
            verification_type = args.get('verification_type', 'both')
            reasoning = args.get('reasoning', '')
            
            # Just acknowledge the function call - actual verification is handled by _process_response
            return f"Verifier function called with verification_type: {verification_type}, reasoning: {reasoning}"
            
        except Exception as e:
            return f"Error in verifier tool: {str(e)}"

