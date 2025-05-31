# By lllyasviel - Mac M 晶片相容版本

import torch

cpu = torch.device('cpu')

# Mac M 晶片相容性修改
if torch.cuda.is_available():
    gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
    print("🖥️ 使用 CUDA GPU")
elif torch.backends.mps.is_available():
    gpu = torch.device('mps')  # Mac M 晶片使用 MPS
    print("🍎 使用 Mac MPS 後端")
else:
    gpu = torch.device('cpu')
    print("⚠️ 使用 CPU，效能可能較慢")

gpu_complete_modules = []

class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return

def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return

def get_cuda_free_memory_gb(device=None):
    """取得可用記憶體（Mac 相容版本）"""
    if device is None:
        device = gpu

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            memory_stats = torch.cuda.memory_stats(device)
            bytes_active = memory_stats['active_bytes.all.current']
            bytes_reserved = memory_stats['reserved_bytes.all.current']
            bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
            bytes_inactive_reserved = bytes_reserved - bytes_active
            bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
            return bytes_total_available / (1024 ** 3)
        except:
            return 16.0  # 預設值
    elif device.type == 'mps':
        # Mac MPS 沒有直接的記憶體查詢，返回預估值
        try:
            import psutil
            virtual_memory = psutil.virtual_memory()
            # 假設可用系統記憶體的 70% 可供 MPS 使用
            available_gb = (virtual_memory.available * 0.7) / (1024 ** 3)
            return min(available_gb, 32)  # MPS 最大約 32GB（統一記憶體）
        except ImportError:
            return 16.0  # 預設值，適合大多數 Mac
    else:
        # CPU 模式
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            return 8.0  # 預設值

def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        current_free_memory = get_cuda_free_memory_gb(target_device)
        if current_free_memory <= preserved_memory_gb:
            _clear_cache(target_device)
            return

        if hasattr(m, 'weight'):
            try:
                m.to(device=target_device)
            except Exception as e:
                print(f"⚠️ 無法移動模組到 {target_device}: {e}")
                break

    try:
        model.to(device=target_device)
    except Exception as e:
        print(f"⚠️ 無法移動模型到 {target_device}: {e}")
    
    _clear_cache(target_device)
    return

def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        current_free_memory = get_cuda_free_memory_gb(target_device)
        if current_free_memory >= preserved_memory_gb:
            _clear_cache(target_device)
            return

        if hasattr(m, 'weight'):
            try:
                m.to(device=cpu)
            except Exception as e:
                print(f"⚠️ 無法卸載模組: {e}")

    try:
        model.to(device=cpu)
    except Exception as e:
        print(f"⚠️ 無法卸載模型: {e}")
    
    _clear_cache(target_device)
    return

def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        try:
            m.to(device=cpu)
            print(f'Unloaded {m.__class__.__name__} as complete.')
        except Exception as e:
            print(f"⚠️ 無法卸載模型 {m.__class__.__name__}: {e}")

    gpu_complete_modules.clear()
    _clear_cache(gpu)
    return

def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    try:
        model.to(device=target_device)
        print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')
        gpu_complete_modules.append(model)
    except Exception as e:
        print(f"⚠️ 無法載入模型到 {target_device}: {e}")
    
    return

def _clear_cache(device):
    """清理記憶體快取"""
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'mps' and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            # 舊版 PyTorch 可能沒有 mps.empty_cache()
            pass
        except Exception as e:
            print(f"⚠️ MPS 快取清理失敗: {e}")