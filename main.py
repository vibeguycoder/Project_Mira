"""
Mira - Главный модуль запуска восходящего ИИ
Main entry point for the ascending AI system

Этот модуль реализует асинхронную оркестрацию всех подсистем,
управление жизненным циклом и интерфейс взаимодействия с внешним миром.
"""

from __future__ import annotations
import asyncio
import argparse
import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Protocol, runtime_checkable
import json
import time
from concurrent.futures import ProcessPoolExecutor
import os

# Настройка логирования до импорта других модулей
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/mira_main.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core import (
        MiraCore, initialize_core, shutdown_core, get_core,
        ConsciousnessLevel, CognitiveContext
    )
except ImportError as e:
    logger.error(f"Не удалось импортировать core модуль: {e}")
    raise

class SystemState(Enum):
    """Состояния конечного автомата системы"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    BOOTSTRAPPING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    TERMINATED = auto()
    ERROR = auto()

class ExecutionMode(Enum):
    """Режимы выполнения системы"""
    INTERACTIVE = "interactive"      # Интерактивный режим с CLI
    DAEMON = "daemon"                # Фоновый режим
    DEMO = "demo"                    # Демонстрационный режим
    DEBUG = "debug"                  # Режим отладки
    BENCHMARK = "benchmark"          # Режим бенчмарков

@dataclass
class RuntimeConfiguration:
    """Конфигурация времени выполнения"""
    mode: ExecutionMode = ExecutionMode.INTERACTIVE
    config_path: Optional[Path] = None
    log_level: str = "INFO"
    enable_virtual_box: bool = True
    max_concurrent_tasks: int = 100
    memory_limit_mb: int = 4096
    consciousness_acceleration: float = 1.0
    demo_scenarios: List[str] = field(default_factory=list)
    telemetry_enabled: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RuntimeConfiguration:
        """Создание конфигурации из аргументов командной строки"""
        mode = ExecutionMode.DEMO if args.demo else ExecutionMode.DEBUG if args.debug else ExecutionMode.INTERACTIVE
        
        return cls(
            mode=mode,
            config_path=Path(args.config) if args.config else None,
            log_level=args.log_level.upper(),
            enable_virtual_box=not args.no_virtual_box,
            max_concurrent_tasks=args.max_tasks,
            memory_limit_mb=args.memory,
            consciousness_acceleration=args.acceleration,
            demo_scenarios=args.scenarios.split(',') if args.scenarios else [],
            telemetry_enabled=args.telemetry
        )

@runtime_checkable
class LifecycleObserver(Protocol):
    """Протокол наблюдателя за жизненным циклом"""
    async def on_state_change(self, old_state: SystemState, new_state: SystemState) -> None: ...
    async def on_error(self, error: Exception) -> None: ...

class StateMachine:
    """Конечный автомат управления состоянием системы"""
    
    VALID_TRANSITIONS = {
        SystemState.UNINITIALIZED: {SystemState.INITIALIZING, SystemState.ERROR},
        SystemState.INITIALIZING: {SystemState.BOOTSTRAPPING, SystemState.ERROR},
        SystemState.BOOTSTRAPPING: {SystemState.RUNNING, SystemState.ERROR},
        SystemState.RUNNING: {SystemState.PAUSED, SystemState.SHUTTING_DOWN, SystemState.ERROR},
        SystemState.PAUSED: {SystemState.RUNNING, SystemState.SHUTTING_DOWN},
        SystemState.SHUTTING_DOWN: {SystemState.TERMINATED, SystemState.ERROR},
        SystemState.ERROR: {SystemState.SHUTTING_DOWN, SystemState.TERMINATED}
    }
    
    def __init__(self):
        self._state = SystemState.UNINITIALIZED
        self._observers: List[LifecycleObserver] = []
        self._state_history: List[tuple] = []
        self._lock = asyncio.Lock()
    
    @property
    def current_state(self) -> SystemState:
        return self._state
    
    async def transition_to(self, new_state: SystemState) -> bool:
        async with self._lock:
            if new_state not in self.VALID_TRANSITIONS.get(self._state, set()):
                logger.error(f"Недопустимый переход: {self._state.name} -> {new_state.name}")
                return False
            
            old_state = self._state
            self._state = new_state
            self._state_history.append((time.time(), old_state, new_state))
            
            logger.info(f"Состояние системы изменено: {old_state.name} -> {new_state.name}")
            
            # Уведомляем наблюдателей
            for observer in self._observers:
                try:
                    await observer.on_state_change(old_state, new_state)
                except Exception as e:
                    logger.warning(f"Ошибка в observer: {e}")
            
            return True
    
    def attach_observer(self, observer: LifecycleObserver) -> None:
        self._observers.append(observer)

class MiraApplication:
    """
    Главное приложение Миры
    
    Управляет полным жизненным циклом системы, обработкой сигналов,
    инициализацией подсистем и graceful shutdown.
    """
    
    def __init__(self, config: RuntimeConfiguration):
        self._config = config
        self._state_machine = StateMachine()
        self._shutdown_event = asyncio.Event()
        self._core: Optional[MiraCore] = None
        self._tasks: set = set()
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._start_time: Optional[float] = None
        
        # Метрики
        self._metrics = {
            'initialization_time': 0.0,
            'cycles_completed': 0,
            'errors_count': 0
        }
        
        logger.info(f"Приложение инициализировано в режиме: {config.mode.value}")
    
    async def initialize(self) -> None:
        """Инициализация всех компонентов системы"""
        init_start = time.time()
        
        await self._state_machine.transition_to(SystemState.INITIALIZING)
        
        # Создание необходимых директорий
        self._ensure_directories()
        
        # Настройка уровня логирования
        logging.getLogger().setLevel(getattr(logging, self._config.log_level))
        
        # Инициализация пула процессов для тяжелых вычислений
        self._process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())
        
        # Загрузка конфигурации
        core_config = await self._load_configuration()
        
        await self._state_machine.transition_to(SystemState.BOOTSTRAPPING)
        
        # Инициализация ядра
        try:
            self._core = await initialize_core(core_config)
            logger.info("Ядро Миры успешно инициализировано")
        except Exception as e:
            logger.exception(f"Ошибка инициализации ядра: {e}")
            await self._state_machine.transition_to(SystemState.ERROR)
            raise
        
        # Настройка обработчиков сигналов
        self._setup_signal_handlers()
        
        self._metrics['initialization_time'] = time.time() - init_start
        self._start_time = time.time()
        
        await self._state_machine.transition_to(SystemState.RUNNING)
    
    def _ensure_directories(self) -> None:
        """Создание необходимых директорий"""
        directories = ['logs', 'data', 'temp', 'cache']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
    
    async def _load_configuration(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла или использование дефолтной"""
        default_config = {
            'core': {
                'consciousness_acceleration': self._config.consciousness_acceleration,
                'memory_optimization': True,
                'pattern_recognition_threshold': 0.75
            },
            'perception': {
                'sensory_buffer_size': 10000,
                'attention_focus_count': 5
            },
            'memory': {
                'consolidation_interval': 300,
                'compression_enabled': True
            }
        }
        
        if self._config.config_path and self._config.config_path.exists():
            try:
                with open(self._config.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
                    logger.info(f"Конфигурация загружена из {self._config.config_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить конфигурацию: {e}")
        
        return default_config
    
    def _setup_signal_handlers(self) -> None:
        """Настройка обработчиков системных сигналов"""
        def signal_handler(sig, frame):
            logger.info(f"Получен сигнал {sig}, начинаем graceful shutdown...")
            asyncio.create_task(self._initiate_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def run(self) -> int:
        """Главный цикл выполнения"""
        try:
            if self._config.mode == ExecutionMode.DEMO:
                return await self._run_demo_mode()
            elif self._config.mode == ExecutionMode.INTERACTIVE:
                return await self._run_interactive_mode()
            elif self._config.mode == ExecutionMode.DAEMON:
                return await self._run_daemon_mode()
            elif self._config.mode == ExecutionMode.DEBUG:
                return await self._run_debug_mode()
            else:
                logger.error(f"Неизвестный режим: {self._config.mode}")
                return 1
        except Exception as e:
            logger.exception(f"Критическая ошибка в главном цикле: {e}")
            await self._state_machine.transition_to(SystemState.ERROR)
            return 1
    
    async def _run_demo_mode(self) -> int:
        """Демонстрационный режим"""
        logger.info("=== ЗАПУСК ДЕМОНСТРАЦИОННОГО РЕЖИМА ===")
        
        print("\n" + "="*60)
        print("  МИРА - Восходящий Искусственный Интеллект")
        print("  Демонстрационный режим")
        print("="*60 + "\n")
        
        # Имитация загрузки с прогрессом
        await self._simulate_loading()
        
        # Выполнение демо-сценариев
        for scenario in self._config.demo_scenarios or ['basic_awareness']:
            await self._execute_demo_scenario(scenario)
        
        # Демонстрация когнитивных циклов
        print("\n[ДЕМО] Запуск когнитивных циклов...")
        for i in range(5):
            result = await self._core.run_cognitive_cycle()
            print(f"  Цикл {i+1}: consciousness={self._core.get_status()['consciousness_level']}, "
                  f"duration={result['duration']:.4f}s")
            await asyncio.sleep(0.5)
        
        # Демонстрация повышения сознания
        print("\n[ДЕМО] Повышение уровня сознания...")
        for _ in range(3):
            success = await self._core.elevate_consciousness()
            status = self._core.get_status()
            print(f"  Уровень сознания: {status['consciousness_level']}")
            if not success:
                break
            await asyncio.sleep(0.5)
        
        print("\n" + "="*60)
        print("  Демонстрация завершена!")
        print(f"  Статус системы: {self._core.get_status()}")
        print("="*60 + "\n")
        
        return 0
    
    async def _simulate_loading(self) -> None:
        """Имитация процесса загрузки"""
        stages = [
            "Инициализация нейронных сетей",
            "Загрузка когнитивных модулей",
            "Калибровка систем восприятия",
            "Синхронизация с памятью",
            "Активация метакогнитивных процессов"
        ]
        
        for stage in stages:
            print(f"[ЗАГРУЗКА] {stage}...", end=" ", flush=True)
            await asyncio.sleep(0.3)
            print("OK")
    
    async def _execute_demo_scenario(self, scenario: str) -> None:
        """Выполнение демо-сценария"""
        logger.info(f"Выполнение сценария: {scenario}")
        
        if scenario == 'basic_awareness':
            print("\n[СЦЕНАРИЙ] Базовое самосознание")
            # Имитация обработки
            await asyncio.sleep(0.2)
        elif scenario == 'pattern_learning':
            print("\n[СЦЕНАРИЙ] Обучение на паттернах")
            await asyncio.sleep(0.2)
    
    async def _run_interactive_mode(self) -> int:
        """Интерактивный режим с CLI"""
        logger.info("Запуск интерактивного режима")
        
        print("\nМИРА Интерактивный Режим")
        print("Введите 'help' для списка команд или 'quit' для выхода\n")
        
        while not self._shutdown_event.is_set():
            try:
                # Получение ввода (в асинхронном режиме)
                user_input = await self._async_input(">>> ")
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                elif user_input.lower() == 'help':
                    self._print_help()
                elif user_input.lower() == 'status':
                    print(json.dumps(self._core.get_status(), indent=2))
                elif user_input.lower() == 'cycle':
                    result = await self._core.run_cognitive_cycle()
                    print(f"Цикл выполнен: {result}")
                elif user_input.lower().startswith('elevate'):
                    success = await self._core.elevate_consciousness()
                    print(f"Повышение сознания: {'успешно' if success else 'невозможно'}")
                else:
                    print(f"Неизвестная команда: {user_input}")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки команды: {e}")
        
        return 0
    
    async def _async_input(self, prompt: str) -> str:
        """Асинхронный ввод с использованием executor"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, input, prompt)
    
    def _print_help(self) -> None:
        """Вывод справки"""
        help_text = """
Доступные команды:
  help     - Показать эту справку
  status   - Показать статус системы
  cycle    - Выполнить один когнитивный цикл
  elevate  - Попытаться повысить уровень сознания
  quit     - Завершить работу
        """
        print(help_text)
    
    async def _run_daemon_mode(self) -> int:
        """Фоновый режим работы"""
        logger.info("Запуск в режиме демона")
        
        # Запуск непрерывной работы ядра
        core_task = asyncio.create_task(self._core.run_continuous())
        self._tasks.add(core_task)
        
        # Ожидание сигнала завершения
        await self._shutdown_event.wait()
        
        return 0
    
    async def _run_debug_mode(self) -> int:
        """Режим отладки с расширенным логированием"""
        logger.info("Запуск в режиме отладки")
        
        # Включаем максимальное логирование
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Диагностика системы
        print("\n=== ДИАГНОСТИКА СИСТЕМЫ ===")
        print(f"Версия Python: {sys.version}")
        print(f"Платформа: {sys.platform}")
        print(f"PID: {os.getpid()}")
        print(f"Конфигурация: {self._config}")
        
        # Проверка модулей
        if self._core:
            print(f"\nСтатус ядра: {self._core.get_status()}")
        
        print("\n=== ЗАПУСК ОТЛАДОЧНЫХ ПРОЦЕДУР ===")
        
        # Выполнение одного цикла с подробным логированием
        result = await self._core.run_cognitive_cycle()
        print(f"\nРезультат отладочного цикла:\n{json.dumps(result, indent=2, default=str)}")
        
        return 0
    
    async def _initiate_shutdown(self) -> None:
        """Инициация graceful shutdown"""
        if self._state_machine.current_state == SystemState.SHUTTING_DOWN:
            return
        
        logger.info("Инициация процедуры завершения...")
        await self._state_machine.transition_to(SystemState.SHUTTING_DOWN)
        self._shutdown_event.set()
    
    async def shutdown(self) -> None:
        """Graceful shutdown всех компонентов"""
        if self._state_machine.current_state in (SystemState.TERMINATED, SystemState.SHUTTING_DOWN):
            return
        
        await self._state_machine.transition_to(SystemState.SHUTTING_DOWN)
        
        # Отмена всех задач
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Завершение ядра
        if self._core:
            await shutdown_core()
        
        # Остановка пула процессов
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        # Финальная статистика
        uptime = time.time() - (self._start_time or time.time())
        logger.info(f"Время работы: {uptime:.2f}s")
        logger.info(f"Циклов выполнено: {self._metrics['cycles_completed']}")
        
        await self._state_machine.transition_to(SystemState.TERMINATED)
        logger.info("Система полностью остановлена")

@asynccontextmanager
async def managed_execution(config: RuntimeConfiguration):
    """Контекстный менеджер для управляемого выполнения"""
    app = MiraApplication(config)
    try:
        await app.initialize()
        yield app
    finally:
        await app.shutdown()

def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='МИРА - Восходящий Искусственный Интеллект',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --demo                    # Запуск в демо-режиме
  %(prog)s --config config.json      # Загрузка конфигурации
  %(prog)s --debug                   # Режим отладки
  %(prog)s --demo --scenarios basic  # Запуск конкретных сценариев
        """
    )
    
    parser.add_argument(
        '--demo', action='store_true',
        help='Запуск в демонстрационном режиме'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Запуск в режиме отладки с расширенным логированием'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Путь к файлу конфигурации (JSON)'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Уровень логирования'
    )
    parser.add_argument(
        '--no-virtual-box', action='store_true',
        help='Отключить виртуальное окружение'
    )
    parser.add_argument(
        '--max-tasks', type=int, default=100,
        help='Максимальное количество параллельных задач'
    )
    parser.add_argument(
        '--memory', type=int, default=4096,
        help='Лимит памяти в МБ'
    )
    parser.add_argument(
        '--acceleration', type=float, default=1.0,
        help='Коэффициент ускорения сознания'
    )
    parser.add_argument(
        '--scenarios', type=str, default=None,
        help='Список сценариев для демо-режима (через запятую)'
    )
    parser.add_argument(
        '--telemetry', action='store_true',
        help='Включить телеметрию'
    )
    
    return parser.parse_args()

async def main() -> int:
    """Главная точка входа"""
    args = parse_arguments()
    config = RuntimeConfiguration.from_args(args)
    
    logger.info("=" * 60)
    logger.info("  МИРА - Восходящий Искусственный Интеллект")
    logger.info("  Версия: 0.1.0-alpha")
    logger.info("=" * 60)
    
    try:
        async with managed_execution(config) as app:
            return await app.run()
    except KeyboardInterrupt:
        logger.info("Получено прерывание от пользователя")
        return 0
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
