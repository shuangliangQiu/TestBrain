from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage
from ..llm.base import BaseLLMService
from ..knowledge.service import KnowledgeService
from .prompts import TestCaseGeneratorPrompt
from utils.logger_manager import get_logger

class TestCaseGeneratorAgent:
    """测试用例生成Agent"""
    
    def __init__(self, llm_service: BaseLLMService, knowledge_service: KnowledgeService, case_design_methods: List[str], case_categories: List[str]):
        self.llm_service = llm_service
        self.case_design_methods = case_design_methods
        self.case_categories = case_categories
        self.knowledge_service = knowledge_service
        self.prompt = TestCaseGeneratorPrompt()
        self.logger = get_logger(self.__class__.__name__)  # 添加logger
    
    def generate(self, input_text: str, input_type: str = "requirement") -> List[Dict[str, Any]]:
        """生成测试用例"""
        self.logger.info(f"开始生成测试用例,进入生成测试用例的TestCaseGeneratorAgent")
        # 确定输入类型描述
        input_type_desc = "需求描述" if input_type == "requirement" else "代码片段"
        
        # 获取知识上下文
        knowledge_context = self._get_knowledge_context(input_text)
        self.logger.info(f"获取到知识库上下文: \n{'='*50}\n{knowledge_context}\n{'='*50}")
        
        # 处理设计方法和测试类型
        case_design_methods = ",".join(self.case_design_methods) if self.case_design_methods else ""
        case_categories = ",".join(self.case_categories) if self.case_categories else ""
        
        # 使用新的 format_messages 方法获取消息列表
        messages = self.prompt.format_messages(
            requirements=input_text,
            case_design_methods=case_design_methods,
            case_categories=case_categories,
            knowledge_context=knowledge_context
        )
        self.logger.info(f"构建后大模型提示词+用户需求消息: \n{'='*50}\n{messages}\n{'='*50}")
        
        # 调用LLM服务
        try:
            response = self.llm_service.invoke(messages)
            result = response.content
            
            # 解析JSON结果
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0].strip()
            else:
                json_str = result
                
            test_cases = json.loads(json_str)
            valid_test_cases = self._validate_test_cases(test_cases)
            return valid_test_cases
            
        except Exception as e:
            raise ValueError(f"无法解析生成的测试用例: {str(e)}\n原始响应: {result}")
    
    def _get_knowledge_context(self, input_text: str) -> str:
        """获取相关知识上下文"""
        try:
            # 暂时返回空字符串，直到KnowledgeService实现完成
            # return ""
            knowledge = self.knowledge_service.search_relevant_knowledge(input_text)
            if knowledge:
                return f"{knowledge}"
        except Exception as e:
            self.logger.warning(f"获取知识上下文失败: {str(e)}")
        return ""
    
    def _validate_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证测试用例格式，返回合法的测试用例列表
        
        Args:
            test_cases: 原始测试用例列表
            
        Returns:
            合法的测试用例列表
            
        Logs:
            记录被过滤掉的测试用例信息
        """
        valid_test_cases = []
        required_fields = {"description", "test_steps", "expected_results"}
        
        for i, test_case in enumerate(test_cases):
            # 检查必要字段是否存在且非空
            if all(field in test_case and test_case[field] for field in required_fields):
                valid_test_cases.append(test_case)
            else:
                missing_fields = [field for field in required_fields 
                                if field not in test_case or not test_case[field]]
                self.logger.warning(
                    f"测试用例 #{i+1} 因缺少必要字段而被过滤: {', '.join(missing_fields)}"
                )
        
        if not valid_test_cases:
            raise ValueError("没有找到任何合法的测试用例")
            
        self.logger.info(f"共验证 {len(test_cases)} 个测试用例，"
                        f"其中 {len(valid_test_cases)} 个合法")
        
        return valid_test_cases
            
