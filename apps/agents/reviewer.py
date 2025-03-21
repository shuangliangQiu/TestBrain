from typing import Dict, Any, List
import json
import logging

from ..llm.base import BaseLLMService
from ..knowledge.service import KnowledgeService 
from ..core.models import TestCase
from .prompts import TestCaseReviewerPrompt
from langchain_core.messages import SystemMessage, HumanMessage
from utils.logger_manager import get_logger



class TestCaseReviewerAgent:
    """测试用例评审Agent"""
    
    def __init__(self, llm_service: BaseLLMService, knowledge_service: KnowledgeService):
        self.llm_service = llm_service
        self.knowledge_service = knowledge_service
        self.prompt = TestCaseReviewerPrompt()
        self.logger = get_logger(self.__class__.__name__)  # 添加logger

    
    def review(self, test_case: TestCase) -> Dict[str, Any]:
        """评审测试用例"""
        try:
            self.logger.info(f"待评审的测试用例数据为: \n{test_case}")
            # 构造提示词
            # formatted_prompt = self._format_prompt(test_case)
            
            # 创建消息列表
            messages = [
                SystemMessage(content=self.prompt.system_template),
                HumanMessage(content=self.prompt.human_template.format(test_case=test_case))
            ]
            
            # 记录消息内容
            # self.logger.info("评审消息详细信息:")
            # for msg in messages:
            #     self.logger.info(f"消息类型: {type(msg).__name__}")
            #     self.logger.info(f"消息内容:\n{msg.content}")
            #     self.logger.info("="*50)
            
            # 调用LLM服务
            result = self.llm_service.invoke(messages)  # 使用 invoke 方法替代 chat
            
            return result
            
        except Exception as e:
            self.logger.error(f"评审过程出错: {str(e)}", exc_info=True)
            raise Exception(f"评审失败: {str(e)}")

    def _format_prompt(self, test_case):
        """格式化提示词"""
        try:
            prompt = f"""请评审以下测试用例：
            测试用例描述：
            {test_case.description}
            测试步骤：
            {test_case.test_steps}
            预期结果：
            {test_case.expected_results}
            """
            return prompt.strip()
            
        except Exception as e:
            self.logger.error(f"格式化提示词时出错: {str(e)}", exc_info=True)
            raise Exception(f"格式化提示词失败: {str(e)}")
