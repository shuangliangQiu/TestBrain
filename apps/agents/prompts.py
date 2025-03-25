from pathlib import Path
import yaml
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

class PromptTemplateManager:
    """提示词模板管理器"""
    
    def __init__(self):
        """初始化，加载配置文件"""
        config_path = Path(__file__).parent / "prompts_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def get_test_case_generator_prompt(self) -> ChatPromptTemplate:
        """获取测试用例生成的提示词模板"""
        config = self.config['test_case_generator']
        
        # 准备系统消息的变量并格式化模板
        system_vars = {
            'role': config['role'],
            'capabilities': config['capabilities'],
            'test_methods': ', '.join(config['test_methods']),
            'test_types': ', '.join(config['test_types'])
        }
        
        # 创建系统消息模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            config['system_template'].format(**system_vars)  # 直接格式化模板
        )
        
        # 创建人类消息模板
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            config['human_template']
        )
        
        # 组合成聊天提示词模板
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def get_test_case_reviewer_prompt(self) -> ChatPromptTemplate:
        """获取测试用例评审的提示词模板"""
        config = self.config['test_case_reviewer']
        
        # 准备系统消息的变量并格式化模板
        system_vars = {
            'role': config['role'],
            'evaluation_aspects': ', '.join(config['evaluation_aspects'])
        }
        
        # 创建系统消息模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            config['system_template'].format(**system_vars)  # 直接格式化模板
        )
        
        # 准备人类消息的变量
        human_vars = {
            'review_points': '\n'.join(f"- {point}" for point in config['review_points'])
        }
        
        # 创建人类消息模板 - 不要在这里格式化 test_case
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            config['human_template']
        )
        
        # 组合成聊天提示词模板
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

class TestCaseGeneratorPrompt:
    """测试用例生成提示词"""
    
    def __init__(self):
        self.prompt_manager = PromptTemplateManager()
        self.prompt_template = self.prompt_manager.get_test_case_generator_prompt()
    
    def format_messages(self, requirements: str, case_design_methods: str = "", 
                       case_categories: str = "", knowledge_context: str = "") -> list:
        """格式化消息
        
        Args:
            requirements: 需求描述
            case_design_methods: 测试用例设计方法
            case_categories: 测试用例类型
            knowledge_context: 知识库上下文
            
        Returns:
            格式化后的消息列表
        """
        # 处理空值情况
        if not case_design_methods:
            case_design_methods = "所有适用的测试用例设计方法"
        
        if not case_categories:
            case_categories = "所有适用的测试类型"
            
        # 格式化知识上下文提示
        knowledge_prompt = (
            f"参考以下知识库内容：\n{knowledge_context}"
            if knowledge_context
            else "根据你的专业知识"
        )
        
        return self.prompt_template.format_messages(
            requirements=requirements,
            case_design_methods=case_design_methods,
            case_categories=case_categories,
            knowledge_context=knowledge_prompt
        )

class TestCaseReviewerPrompt:
    """测试用例评审提示词"""
    
    def __init__(self):
        self.prompt_manager = PromptTemplateManager()
        self.prompt_template = self.prompt_manager.get_test_case_reviewer_prompt()
    
    def format_messages(self, test_case: Dict[str, Any]) -> list:
        """格式化消息
        
        Args:
            test_case: 测试用例数据
            
        Returns:
            格式化后的消息列表
        """
        # 格式化测试用例数据为字符串
        test_case_str = (
            f"测试用例描述：\n{test_case.get('description', '')}\n\n"
            f"测试步骤：\n{test_case.get('test_steps', '')}\n\n"
            f"预期结果：\n{test_case.get('expected_results', '')}"
        )
        
        # 获取评审点列表
        review_points = '\n'.join(
            f"- {point}" 
            for point in self.prompt_manager.config['test_case_reviewer']['review_points']
        )
        
        return self.prompt_template.format_messages(
            test_case=test_case_str,
            review_points=review_points
        )

# 使用示例
if __name__ == "__main__":
    # 测试用例生成
    generator = TestCaseGeneratorPrompt()
    messages = generator.format_messages(
        requirements="实现用户登录功能",
        case_design_methods="等价类划分法",
        case_categories="功能测试",
        knowledge_context="用户登录需要验证用户名和密码"
    )
    print("Generator Messages:", messages)
    
    # 测试用例评审
    reviewer = TestCaseReviewerPrompt()
    test_case = {
        "description": "测试用户登录功能",
        "test_steps": ["1. 输入用户名", "2. 输入密码", "3. 点击登录按钮"],
        "expected_results": ["1. 显示输入框", "2. 密码显示为星号", "3. 登录成功"]
    }
    messages = reviewer.format_messages(test_case)
    print("\nReviewer Messages:", messages)
