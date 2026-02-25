"""
Тесты модуля ядра Миры
Unit tests for Mira Core module

Этот модуль содержит комплексные тесты для проверки
корректности работы когнитивных компонентов системы.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import sys
from pathlib import Path

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    MiraCore, ConsciousnessLevel, CognitiveContext, WorkingMemory,
    PerceptionModule, MemoryModule, ReasoningModule, TemporalState,
    AbstractCognitiveModule, PatternRecognizer, PatternRecognitionResult,
    initialize_core, shutdown_core, get_core, CognitivePriority,
    BehaviorManifestation, CognitiveRegistry
)


# ============ Fixtures ============

@pytest.fixture
def event_loop():
    """Создание event loop для async тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def cognitive_context():
    """Фикстура для когнитивного контекста"""
    return CognitiveContext(
        session_id="test_session_001",
        consciousness_level=ConsciousnessLevel.REACTIVE,
        working_memory=WorkingMemory(capacity=100),
        long_term_refs=set(),
        temporal_state=TemporalState(
            current_cycle=42,
            time_delta=1.5,
            continuity_index=0.95
        )
    )


@pytest.fixture
async def initialized_core():
    """Фикстура инициализированного ядра"""
    core = MiraCore({
        'perception': {'test_mode': True},
        'memory': {'consolidation_enabled': False}
    })
    await core.bootstrap()
    yield core
    await core.shutdown()


# ============ Тесты базовых компонентов ============

class TestWorkingMemory:
    """Тесты рабочей памяти"""
    
    @pytest.mark.asyncio
    async def test_memory_initialization(self):
        """Тест инициализации рабочей памяти"""
        memory = WorkingMemory(capacity=500)
        assert memory._capacity == 500
        assert len(memory._buffer) == 0
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self):
        """Тест сохранения и извлечения данных"""
        memory = WorkingMemory(capacity=10)
        
        # Сохраняем данные
        await memory.store("key1", {"value": 42}, CognitivePriority.HIGH)
        await memory.store("key2", {"value": 100}, CognitivePriority.NORMAL)
        
        # Проверяем извлечение
        data1 = await memory.retrieve("key1")
        assert data1 is not None
        assert data1["value"] == 42
        
        data2 = await memory.retrieve("key2")
        assert data2["value"] == 100
    
    @pytest.mark.asyncio
    async def test_memory_capacity_limit(self):
        """Тест ограничения емкости памяти"""
        memory = WorkingMemory(capacity=3)
        
        # Заполняем память
        for i in range(5):
            await memory.store(f"key_{i}", {"idx": i})
        
        # Проверяем что старые данные вытеснены
        assert len(memory._buffer) <= 3
    
    @pytest.mark.asyncio
    async def test_memory_consolidation_candidates(self):
        """Тест выявления кандидатов на консолидацию"""
        memory = WorkingMemory(capacity=100)
        
        await memory.store("frequent", {"data": 1})
        
        # Многократный доступ
        for _ in range(10):
            await memory.retrieve("frequent")
        
        candidates = await memory.consolidate()
        assert "frequent" in candidates


class TestTemporalState:
    """Тесты временного состояния"""
    
    def test_temporal_state_creation(self):
        """Тест создания временного состояния"""
        state = TemporalState(
            current_cycle=100,
            time_delta=2.5,
            continuity_index=0.98
        )
        
        assert state.current_cycle == 100
        assert state.time_delta == 2.5
        assert state.continuity_index == 0.98
    
    def test_temporal_state_continuity_range(self):
        """Тест диапазона непрерывности"""
        # Проверка что индекс непрерывности в допустимых пределах
        state = TemporalState(1, 0.0, 1.0)
        assert 0.0 <= state.continuity_index <= 1.0


class TestPatternRecognizer:
    """Тесты распознавателя паттернов"""
    
    @pytest.mark.asyncio
    async def test_pattern_recognition(self):
        """Тест базового распознавания паттерна"""
        recognizer = PatternRecognizer("visual_pattern", threshold=0.8)
        
        result = await recognizer.recognize({"features": [1, 2, 3]})
        
        assert isinstance(result, PatternRecognitionResult)
        assert result.pattern_type == "visual_pattern"
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_recognizer_threshold(self):
        """Тест порога распознавания"""
        recognizer = PatternRecognizer("test", threshold=0.5)
        
        # Симуляция распознавания с разными уровнями уверенности
        result = await recognizer.recognize({"weak": True})
        assert result.confidence >= 0.0


class TestCognitiveRegistry:
    """Тесты реестра когнитивных компонентов"""
    
    def test_registry_signatures(self):
        """Тест получения сигнатур компонентов"""
        signatures = CognitiveRegistry.get_all_signatures()
        assert isinstance(signatures, dict)
    
    def test_component_registration(self):
        """Тест регистрации компонентов"""
        # Проверяем что AbstractCognitiveModule зарегистрирован
        assert "AbstractCognitiveModule" in CognitiveRegistry._registry


class TestConsciousnessLevel:
    """Тесты уровней сознания"""
    
    def test_consciousness_progression(self):
        """Тест прогрессии уровней сознания"""
        levels = list(ConsciousnessLevel)
        assert ConsciousnessLevel.DORMANT in levels
        assert ConsciousnessLevel.ASCENDING in levels
        assert len(levels) == 5
    
    def test_consciousness_ordering(self):
        """Тест порядка уровней сознания"""
        # Проверка что порядок соответствует развитию
        assert ConsciousnessLevel.DORMANT.value < ConsciousnessLevel.ASCENDING.value


# ============ Тесты модулей ============

class TestPerceptionModule:
    """Тесты модуля восприятия"""
    
    @pytest.mark.asyncio
    async def test_perception_initialization(self):
        """Тест инициализации модуля восприятия"""
        module = PerceptionModule({"test": True})
        await module.initialize()
        
        assert module.is_active == False  # Не запущен до start_processing_loop
        assert "visual" in module._sensory_buffer
    
    @pytest.mark.asyncio
    async def test_perception_processing(self, cognitive_context):
        """Тест обработки сенсорного ввода"""
        module = PerceptionModule({})
        await module.initialize()
        
        result = await module.process({
            "modality": "visual",
            "data": {"pixels": [[1, 2], [3, 4]]},
            "novelty": 0.8,
            "intensity": 0.9
        }, cognitive_context)
        
        assert result["modality"] == "visual"
        assert "salience" in result
        assert 0 <= result["salience"] <= 1
    
    @pytest.mark.asyncio
    async def test_salience_calculation(self):
        """Тест расчета заметности"""
        module = PerceptionModule({})
        
        salience = module._calculate_salience({
            "novelty": 1.0,
            "intensity": 1.0
        })
        
        assert salience > 0.9  # Высокая заметность


class TestMemoryModule:
    """Тесты модуля памяти"""
    
    @pytest.mark.asyncio
    async def test_memory_initialization(self):
        """Тест инициализации модуля памяти"""
        module = MemoryModule({"test": True})
        await module.initialize()
        
        assert module._episodic_store == {}
        assert module._consolidation_task is not None
    
    @pytest.mark.asyncio
    async def test_episodic_storage(self, cognitive_context):
        """Тест эпизодического хранения"""
        module = MemoryModule({})
        await module.initialize()
        
        memory_id = await module.process({
            "operation": "store_episodic",
            "data": {"event": "test_event", "concepts": ["test"], "emotional_valence": 0.5}
        }, cognitive_context)
        
        assert memory_id.startswith("ep_")
        assert memory_id in module._episodic_store
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, cognitive_context):
        """Тест извлечения из памяти"""
        module = MemoryModule({})
        await module.initialize()
        
        # Сохраняем тестовые данные
        await module._store_episodic({
            "concepts": ["cat", "animal", "pet"],
            "description": "Test memory"
        }, cognitive_context)
        
        # Ищем по концептам
        results = await module._retrieve({
            "concepts": ["cat", "pet"]
        }, cognitive_context)
        
        assert isinstance(results, list)
        # Проверка что результаты отсортированы
        if len(results) > 1:
            assert results[0]["relevance"] >= results[-1]["relevance"]
    
    @pytest.mark.asyncio
    async def test_semantic_association(self):
        """Тест семантической ассоциации"""
        module = MemoryModule({})
        
        result = await module._create_association("dog", "animal")
        assert result == True
        assert "animal" in module._semantic_network.get("dog", set())


class TestReasoningModule:
    """Тесты модуля рассуждения"""
    
    @pytest.mark.asyncio
    async def test_reasoning_initialization(self):
        """Тест инициализации модуля рассуждения"""
        module = ReasoningModule({"test": True})
        await module.initialize()
        
        assert module._goal_stack == []
    
    @pytest.mark.asyncio
    async def test_deductive_inference(self, cognitive_context):
        """Тест дедуктивного вывода"""
        module = ReasoningModule({})
        await module.initialize()
        
        result = await module.process({
            "type": "deduce",
            "premises": ["Все люди смертны", "Сократ - человек"],
            "conclusion": "Сократ смертен"
        }, cognitive_context)
        
        assert result["inference_type"] == "deductive"
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_plan_creation(self, cognitive_context):
        """Тест создания плана"""
        module = ReasoningModule({})
        await module.initialize()
        
        result = await module.process({
            "type": "plan",
            "goal": "достичь цели X",
            "constraints": ["время < 1 часа"]
        }, cognitive_context)
        
        assert result["goal"] == "достичь цели X"
        assert "steps" in result
    
    @pytest.mark.asyncio
    async def test_situation_evaluation(self, cognitive_context):
        """Тест оценки ситуации"""
        module = ReasoningModule({})
        await module.initialize()
        
        result = await module.process({
            "type": "evaluate",
            "situation": {"danger_level": 0.3, "opportunity": 0.7}
        }, cognitive_context)
        
        assert "risk_assessment" in result
        assert "opportunity_assessment" in result
        assert 0 <= result["risk_assessment"] <= 1


# ============ Тесты интеграции ============

class TestMiraCoreIntegration:
    """Интеграционные тесты ядра Миры"""
    
    @pytest.mark.asyncio
    async def test_core_initialization(self):
        """Тест инициализации ядра"""
        core = MiraCore({"test": True})
        
        assert core.get_status()["consciousness_level"] == "DORMANT"
        assert core.get_status()["cycle_count"] == 0
    
    @pytest.mark.asyncio
    async def test_core_bootstrap(self):
        """Тест загрузки ядра"""
        core = MiraCore({})
        await core.bootstrap()
        
        status = core.get_status()
        assert status["consciousness_level"] in ["REACTIVE", "ADAPTIVE"]
        assert status["modules_active"] >= 3
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_execution(self):
        """Тест выполнения когнитивного цикла"""
        core = MiraCore({})
        await core.bootstrap()
        
        result = await core.run_cognitive_cycle()
        
        assert result["cycle"] == 1
        assert "duration" in result
        assert "results" in result
        assert result["duration"] >= 0
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        """Тест выполнения нескольких циклов"""
        core = MiraCore({})
        await core.bootstrap()
        
        results = []
        for _ in range(5):
            result = await core.run_cognitive_cycle()
            results.append(result)
        
        # Проверка последовательности циклов
        for i, result in enumerate(results):
            assert result["cycle"] == i + 1
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_consciousness_elevation(self):
        """Тест повышения уровня сознания"""
        core = MiraCore({})
        await core.bootstrap()
        
        initial_level = core.get_status()["consciousness_level"]
        
        # Пытаемся повысить сознание
        success = await core.elevate_consciousness()
        
        if success:
            new_level = core.get_status()["consciousness_level"]
            assert new_level != initial_level
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_consciousness_max_level(self):
        """Тест максимального уровня сознания"""
        core = MiraCore({})
        await core.bootstrap()
        
        # Повышаем до максимума
        for _ in range(10):
            await core.elevate_consciousness()
        
        status = core.get_status()
        # Должны достичь ASCENDING
        assert status["consciousness_level"] == "ASCENDING"
        
        # Дальше повысить нельзя
        success = await core.elevate_consciousness()
        assert success == False
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_input_processing(self):
        """Тест обработки ввода"""
        core = MiraCore({})
        await core.bootstrap()
        
        result = await core.process_input({
            "type": "sensor_data",
            "modality": "test",
            "data": {"value": 123}
        })
        
        assert "perception" in result
        assert "reasoning" in result
        assert "timestamp" in result
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_core_status(self):
        """Тест получения статуса системы"""
        core = MiraCore({})
        await core.bootstrap()
        
        # Выполняем несколько циклов
        for _ in range(3):
            await core.run_cognitive_cycle()
        
        status = core.get_status()
        
        assert "consciousness_level" in status
        assert "cycle_count" in status
        assert "uptime" in status
        assert status["cycle_count"] == 3
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Тест корректного завершения"""
        core = MiraCore({})
        await core.bootstrap()
        
        # Выполняем некоторую работу
        await core.run_cognitive_cycle()
        
        await core.shutdown()
        
        # Проверяем что ядро корректно остановлено
        status = core.get_status()
        assert status["modules_active"] == 0


class TestGlobalFunctions:
    """Тесты глобальных функций"""
    
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Тест инициализации и завершения через глобальные функции"""
        core = await initialize_core({"test": True})
        assert core is not None
        
        global_core = get_core()
        assert global_core is core
        
        await shutdown_core()
        assert get_core() is None


# ============ Тесты производительности ============

class TestPerformance:
    """Тесты производительности"""
    
    @pytest.mark.asyncio
    async def test_cycle_performance(self):
        """Тест производительности когнитивного цикла"""
        core = MiraCore({})
        await core.bootstrap()
        
        start = time.time()
        await core.run_cognitive_cycle()
        duration = time.time() - start
        
        # Цикл должен выполняться быстро (менее 5 секунд для пустой операции)
        assert duration < 5.0
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Тест под нагрузкой памяти"""
        memory = WorkingMemory(capacity=10000)
        
        start = time.time()
        for i in range(1000):
            await memory.store(f"key_{i}", {"data": i * 1000})
        
        duration = time.time() - start
        
        # Должно выполняться быстро
        assert duration < 10.0
        assert len(memory._buffer) <= 10000


# ============ Тесты edge cases ============

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    @pytest.mark.asyncio
    async def test_empty_input_processing(self):
        """Тест обработки пустого ввода"""
        core = MiraCore({})
        await core.bootstrap()
        
        result = await core.process_input({})
        assert result is not None
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_rapid_consciousness_changes(self):
        """Тест быстрых изменений сознания"""
        core = MiraCore({})
        await core.bootstrap()
        
        # Быстрое чередование
        for _ in range(20):
            await core.elevate_consciousness()
        
        status = core.get_status()
        # Должно быть стабильное состояние
        assert status["consciousness_level"] in [level.name for level in ConsciousnessLevel]
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self):
        """Тест конкурентного доступа к памяти"""
        memory = WorkingMemory(capacity=1000)
        
        async def writer(idx):
            for i in range(10):
                await memory.store(f"concurrent_{idx}_{i}", {"idx": idx, "i": i})
        
        # Запускаем несколько писателей
        await asyncio.gather(*[writer(i) for i in range(5)])
        
        # Проверяем целостность
        assert len(memory._buffer) <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
