import yaml
from pathlib import Path
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class BasePrompt:
    """提示词基类"""
    @property
    def system_template(self) -> str:
        """系统提示模板"""
        raise NotImplementedError("子类必须实现system_template方法")
    
    @property
    def human_template(self) -> str:
        """人类提示模板"""
        raise NotImplementedError("子类必须实现human_template方法")
    
    def get_chat_prompt(self) -> ChatPromptTemplate:
        """获取聊天提示模板"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])


class TestCaseGeneratorPrompt(BasePrompt):
    """测试用例生成提示词"""
    
    def _load_config(self):
        config_path = Path(__file__).parent / "prompts_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["test_case_generator"]
    
    @property
    def system_template(self) -> str:
        return self._load_config()["system_template"]
    
    @property
    def human_template(self) -> str:
        return self._load_config()["human_template"]


class TestCaseReviewerPrompt(BasePrompt):
    """测试用例评审提示词"""
    
    def _load_config(self):
        config_path = Path(__file__).parent / "prompts_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["test_case_reviewer"]
    
    @property
    def system_template(self) -> str:
        return self._load_config()["system_template"]
    
    @property
    def human_template(self) -> str:
        return self._load_config()["human_template"]
