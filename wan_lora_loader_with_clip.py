import os
import copy
import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import comfy.sd
import gc
import threading
import hashlib
import math
from .lora_utils import parse_wan_lora

_object_patches_lock = threading.RLock()
_tensor_hash_lock = threading.Lock()

def _parse_env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        print(f"Warning: Invalid value for {name}, using default: {default}")
        return default

def _parse_env_bool(name, default="0"):
    val = os.environ.get(name, default)
    return val == "1"

_ENABLE_CUDA_SYNC = _parse_env_bool("INT8_ENABLE_CUDA_SYNC", "0")
_CLEAR_CACHE_STRATEGY = os.environ.get("INT8_CLEAR_CACHE", "auto")
_MAX_LORA_CACHE_SIZE = _parse_env_int("INT8_LORA_CACHE_SIZE", 8)

def _aggressive_memory_cleanup():
    """Aggressive memory cleanup for OOM situations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

_MAX_FULL_HASH_ELEMENTS = 64
_HASH_SAMPLE_COUNT = 16

def _should_clear_cache():
    if _CLEAR_CACHE_STRATEGY == "always":
        return True
    if _CLEAR_CACHE_STRATEGY == "never":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved > 0:
            utilization = allocated / reserved
            return utilization < 0.8
        if allocated > 0:
            return False
    except Exception:
        pass
    return True


def _get_tensor_content_hash(t, full_hash_threshold=1024*1024):
    """Get a stable hash for tensor identity based on content."""
    if not isinstance(t, torch.Tensor):
        return (id(t),)

    shape_str = str(t.shape)
    dtype_str = str(t.dtype)
    numel = t.numel()
    
    if numel == 0:
        return (shape_str, dtype_str, 0)

    if _ENABLE_CUDA_SYNC and torch.cuda.is_available() and t.is_cuda:
        torch.cuda.synchronize(t.device)

    with _tensor_hash_lock:
        try:
            flat = t.flatten().detach()
            tensor_bytes = numel * t.element_size()
            
            if tensor_bytes <= full_hash_threshold:
                cpu_tensor = flat.cpu()
                content_bytes = cpu_tensor.numpy().tobytes()
                content_hash = hashlib.sha256(content_bytes).hexdigest()
                return (shape_str, dtype_str, numel, content_hash)
            else:
                num_samples = min(256, numel)
                
                bucket_size = numel // num_samples
                indices = []
                for i in range(num_samples):
                    idx = i * bucket_size + bucket_size // 2
                    indices.append(min(idx, numel - 1))
                
                indices.extend([0, numel - 1, numel // 2, numel // 4, 3 * numel // 4])
                indices = sorted(set(indices))
                
                sample = flat[indices].cpu()
                sample_bytes = sample.numpy().tobytes()
                sample_hash = hashlib.sha256(sample_bytes).hexdigest()
                
                flat_f64 = flat.to(torch.float64)
                tensor_sum = float(flat_f64.sum().item())
                tensor_abs_sum = float(flat_f64.abs().sum().item())
                tensor_min = float(flat_f64.min().item())
                tensor_max = float(flat_f64.max().item())
                
                return (shape_str, dtype_str, numel, sample_hash,
                        tensor_sum, tensor_abs_sum, tensor_min, tensor_max)
                
        except Exception as e:
            return (shape_str, dtype_str, numel, id(t), "fallback", str(type(e).__name__))


def _get_patch_identity(patches):
    """Get a stable identity for patches list that avoids id() reuse issues."""
    if not patches:
        return ()
    
    try:
        patches_snapshot = tuple(patches)
    except (RuntimeError, TypeError):
        import time
        return ("unstable", id(patches), time.time())
    
    identities = []
    for patch in patches_snapshot:
        if isinstance(patch, (list, tuple)) and len(patch) >= 2:
            d, u = patch[0], patch[1]
            patch_id = (
                _get_tensor_content_hash(d),
                _get_tensor_content_hash(u),
                len(patch)
            )
            if len(patch) >= 3:
                alpha = patch[2]
                if isinstance(alpha, (int, float)):
                    patch_id = patch_id + (alpha,)
                elif isinstance(alpha, torch.Tensor):
                    patch_id = patch_id + (float(alpha.item()),)
            identities.append(patch_id)
        else:
            identities.append((type(patch).__name__, id(patch)))
    return tuple(identities)


def _is_valid_alpha(a):
    """Check if alpha value is valid (not None, NaN, or zero)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        try:
            a_val = a.item()
            return not (math.isnan(a_val) or a_val == 0)
        except:
            return False
    
    if isinstance(a, (int, float)):
        return not (a != a or a == 0)
    
    return False


def _is_zero_alpha(a):
    """Check if alpha is exactly zero (for warning suppression)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        try:
            return a.item() == 0
        except:
            return False
    
    return a == 0


def _identify_qkv_component(key: str) -> tuple:
    """Identify which QKV component a LoRA key targets."""
    key_lower = key.lower()
    
    fused_patterns = ['.qkv', '_qkv', '.to_qkv', '_to_qkv']
    for pattern in fused_patterns:
        if pattern in key_lower:
            return (None, True)
    
    q_patterns = ['.to_q.', '.q_proj.', '_q.', '.q.', '_attn_q', '/q/']
    k_patterns = ['.to_k.', '.k_proj.', '_k.', '.k.', '_attn_k', '/k/']
    v_patterns = ['.to_v.', '.v_proj.', '_v.', '.v.', '_attn_v', '/v/']
    
    for pattern in q_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('q', False)
    
    for pattern in k_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('k', False)
    
    for pattern in v_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('v', False)
    
    return (None, False)


def _validate_lora_bounds(offset, size, out_dim, key=""):
    """Validate and adjust LoRA patch bounds to fit output tensor."""
    if offset < 0:
        print(f"Warning: Negative offset {offset} for {key}, clamping to 0")
        offset = 0
    
    if size < 0:
        print(f"Warning: Negative size {size} for {key}, treating as full size")
        size = 0
    
    if size == 0:
        size = out_dim - offset
    
    if offset >= out_dim:
        print(f"Warning: Offset {offset} >= output dim {out_dim} for {key}, skipping")
        return 0, 0, False
    
    if offset + size > out_dim:
        old_size = size
        size = out_dim - offset
        print(f"Warning: Clamping size from {old_size} to {size} for {key}")
    
    if size <= 0:
        return 0, 0, False
    
    return offset, size, True


class LoRAWeightCache:
    """Thread-safe LRU cache for LoRA weights with memory awareness."""

    def __init__(self, max_size=_MAX_LORA_CACHE_SIZE):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
        self._lock = threading.RLock()
        self._ref_counts = {}

    def _check_memory_pressure(self):
        """Check if GPU memory is under pressure."""
        if not torch.cuda.is_available():
            return False
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem < total_mem * 0.15
        except Exception:
            return False

    def _evict_for_memory(self):
        """Evict entries to free GPU memory."""
        evicted = 0
        for key in list(self._access_order):
            if self._ref_counts.get(key, 0) == 0:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                if key in self._cache:
                    value = self._cache.pop(key)
                    if isinstance(value, dict):
                        for v in value.values():
                            if isinstance(v, torch.Tensor):
                                del v
                    elif isinstance(value, torch.Tensor):
                        del value
                    evicted += 1
                self._ref_counts.pop(key, None)
                
                if evicted >= 2 and not self._check_memory_pressure():
                    break
        
        if evicted > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return evicted

    def get(self, key):
        """Get cached value directly (no clone for performance)."""
        with self._lock:
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                return self._cache[key]
            return None

    def get_or_create(self, key, factory_fn):
        """Atomically get existing value or create and cache new one."""
        with self._lock:
            if self._check_memory_pressure():
                self._evict_for_memory()
            
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                return (self._cache[key], False)
            
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            value = factory_fn()
            
            self._cache[key] = value
            self._access_order.append(key)
            self._ref_counts[key] = 1
            
            return (value, True)

    def release(self, key):
        """Release a reference obtained via get()."""
        with self._lock:
            if key in self._ref_counts:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    del self._ref_counts[key]
                    if self._check_memory_pressure() and key in self._cache:
                        try:
                            self._access_order.remove(key)
                        except ValueError:
                            pass
                        value = self._cache.pop(key, None)
                        if value is not None:
                            if isinstance(value, dict):
                                for v in value.values():
                                    if isinstance(v, torch.Tensor):
                                        del v
                            elif isinstance(value, torch.Tensor):
                                del value

    def _evict_oldest(self):
        """Evict oldest entry with zero reference count."""
        for key in list(self._access_order):
            if self._ref_counts.get(key, 0) == 0:
                self._access_order.remove(key)
                if key in self._cache:
                    self._cache.pop(key)
                return True
        return False

    def clear(self):
        """Clear cache entries with zero reference count."""
        with self._lock:
            keys_to_evict = [k for k in self._access_order
                            if self._ref_counts.get(k, 0) == 0]
            
            for key in keys_to_evict:
                if key in self._cache:
                    value = self._cache.pop(key)
                    if isinstance(value, dict):
                        for v in value.values():
                            if isinstance(v, torch.Tensor):
                                del v
                    elif isinstance(value, torch.Tensor):
                        del value
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._ref_counts.pop(key, None)
            
            if keys_to_evict:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class LoRAWrapperModule(torch.nn.Module):
    def __init__(self, wrapped_module, lora_patches_list):
        super().__init__()
        if lora_patches_list is not None and not isinstance(lora_patches_list, (list, tuple)):
            raise TypeError(f"lora_patches_list must be a list or tuple, got {type(lora_patches_list).__name__}")
        
        self._getting_attr = threading.local()
        self._getattr_lock = threading.Lock()
        
        self._wrapped_module = wrapped_module
        self.lora_patches = lora_patches_list if lora_patches_list is not None else []
        self._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self._last_patch_ids = None
        self._patch_ids_lock = threading.Lock()

    def __setstate__(self, state):
        """Restore object state from pickle, handling backwards compatibility."""
        torch.nn.Module.__init__(self)
        
        if 'wrapped_module' in state and '_wrapped_module' not in state:
            state['_wrapped_module'] = state.pop('wrapped_module')
        
        if '_wrapped_module' not in state:
            state['_wrapped_module'] = None
        
        wrapped = state.pop('_wrapped_module', None)
        lora_patches = state.pop('lora_patches', [])
        
        state.pop('_getting_attr', None)
        state.pop('_getattr_lock', None)
        state.pop('_patch_ids_lock', None)
        state.pop('_lora_weight_cache', None)
        state.pop('_last_patch_ids', None)
        
        self.__dict__.update(state)
        
        self._wrapped_module = wrapped
        self.lora_patches = lora_patches
        
        self._getting_attr = threading.local()
        self._getattr_lock = threading.Lock()
        self._patch_ids_lock = threading.Lock()
        self._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self._last_patch_ids = None

    @property
    def wrapped_module(self):
        if '_wrapped_module' in self.__dict__:
            return self.__dict__['_wrapped_module']
        if 'wrapped_module' in self.__dict__:
            return self.__dict__['wrapped_module']
        raise AttributeError("wrapped_module not set")
    
    @wrapped_module.setter
    def wrapped_module(self, value):
        self._wrapped_module = value

    @property
    def weight(self):
        """Expose weight directly for ComfyUI compatibility."""
        wrapped = self.wrapped_module
        if wrapped is not None and hasattr(wrapped, 'weight'):
            return wrapped.weight
        raise AttributeError("wrapped module has no weight")

    @property
    def bias(self):
        """Expose bias directly for ComfyUI compatibility."""
        wrapped = self.wrapped_module
        if wrapped is not None and hasattr(wrapped, 'bias'):
            return wrapped.bias
        return None

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles threading.local() and locks."""
        new_instance = object.__new__(self.__class__)
        memo[id(self)] = new_instance
        
        torch.nn.Module.__init__(new_instance)
        
        wrapped = self.wrapped_module
        if wrapped is not None:
            new_instance._wrapped_module = copy.deepcopy(wrapped, memo)
        else:
            new_instance._wrapped_module = None
        
        new_instance.lora_patches = copy.deepcopy(self.lora_patches, memo)
        
        new_instance._getting_attr = threading.local()
        new_instance._getattr_lock = threading.Lock()
        new_instance._patch_ids_lock = threading.Lock()
        
        new_instance._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        new_instance._last_patch_ids = None
        
        return new_instance

    def __reduce__(self):
        """Support proper pickling by providing reconstruction info."""
        wrapped = self.wrapped_module
        patches = self.lora_patches
        return (
            self.__class__,
            (wrapped, patches),
        )

    def _is_getting_attr(self, name):
        """Check if we're already getting this attribute (per-thread)."""
        try:
            return name in self._getting_attr.set
        except AttributeError:
            return False

    def _add_getting_attr(self, name):
        """Add to getting_attr set (per-thread)."""
        try:
            self._getting_attr.set.add(name)
        except AttributeError:
            self._getting_attr.set = {name}

    def _remove_getting_attr(self, name):
        """Remove from getting_attr set (per-thread)."""
        try:
            self._getting_attr.set.discard(name)
        except AttributeError:
            pass

    def __getattr__(self, name):
        if name in ('_wrapped_module', 'wrapped_module'):
            if '_wrapped_module' in self.__dict__:
                return self.__dict__['_wrapped_module']
            if 'wrapped_module' in self.__dict__:
                return self.__dict__['wrapped_module']
            if '_modules' in self.__dict__:
                if 'wrapped_module' in self._modules:
                    return self._modules['wrapped_module']
                if '_wrapped_module' in self._modules:
                    return self._modules['_wrapped_module']
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        try:
            getting_attr = object.__getattribute__(self, '_getting_attr')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if self._is_getting_attr(name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (recursion detected)")
        
        self._add_getting_attr(name)
        try:
            wrapped = None
            if '_wrapped_module' in self.__dict__:
                wrapped = self.__dict__['_wrapped_module']
            elif 'wrapped_module' in self.__dict__:
                wrapped = self.__dict__['wrapped_module']
            
            if wrapped is not None:
                return getattr(wrapped, name)
            
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        finally:
            self._remove_getting_attr(name)

    def __delattr__(self, name):
        """Safe attribute deletion with protected attributes."""
        _protected_attrs = {'_getting_attr', '_getattr_lock', '_patch_ids_lock',
                            '_wrapped_module', 'lora_patches', '_lora_weight_cache'}
        
        if name in _protected_attrs:
            raise AttributeError(f"Cannot delete protected attribute '{name}'")
        
        if name in self.__dict__:
            object.__delattr__(self, name)
            return
        
        if name in self._modules:
            del self._modules[name]
            return
        if name in self._parameters:
            del self._parameters[name]
            return
        if name in self._buffers:
            del self._buffers[name]
            return
        
        try:
            wrapped = self.wrapped_module
        except AttributeError:
            wrapped = None
        if wrapped is not None:
            try:
                delattr(wrapped, name)
                return
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x):
        wrapped = self.wrapped_module
        original_out = wrapped(x)
        
        patches = self.lora_patches
        if not patches:
            return original_out

        try:
            from .int8_quant import chunked_int8_lora_forward, chunked_lora_forward
        except ImportError as e:
            print(f"ERROR: Failed to import int8_quant module: {e}")
            return original_out

        with self._patch_ids_lock:
            current_patch_ids = _get_patch_identity(patches)
            if self._last_patch_ids != current_patch_ids:
                self._lora_weight_cache.clear()
            self._last_patch_ids = current_patch_ids

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        out = None
        keys_to_release = set()
        any_patch_applied = False
        
        try:
            for patch_data in patches:
                if not isinstance(patch_data, (list, tuple)):
                    continue

                patch_len = len(patch_data)
                if patch_len == 3:
                    d, u, a = patch_data
                    d_scale, u_scale = None, None
                    offset, size = 0, 0
                elif patch_len == 5:
                    d, u, a, d_scale, u_scale = patch_data
                    offset, size = 0, 0
                elif patch_len == 7:
                    d, u, a, d_scale, u_scale, offset, size = patch_data
                else:
                    continue

                if not all(isinstance(t, torch.Tensor) for t in [d, u]):
                    continue

                if not _is_valid_alpha(a):
                    continue

                is_int8 = d.dtype == torch.int8 and u.dtype == torch.int8
                expected_input_dim = d.shape[1]
                
                if expected_input_dim != x_2d.shape[-1]:
                    continue

                offset, size, is_valid = _validate_lora_bounds(
                    offset, size, original_out.shape[-1],
                    key=f"patch_{id(patch_data)}"
                )
                if not is_valid:
                    continue

                if out is None:
                    if not original_out.requires_grad and original_out.is_contiguous():
                        out = original_out
                    else:
                        out = original_out.clone()
                
                try:
                    if is_int8 and d_scale is not None and u_scale is not None:
                        cache_key = (
                            "int8",
                            _get_tensor_content_hash(d), _get_tensor_content_hash(u),
                            _get_tensor_content_hash(d_scale), _get_tensor_content_hash(u_scale),
                            offset, size, original_out.shape[-1],
                        )
                        
                        def create_int8_cache():
                            return {
                                'd': d.to(device=original_out.device, non_blocking=True),
                                'u': u.to(device=original_out.device, non_blocking=True),
                                'd_scale': d_scale.to(device=original_out.device, non_blocking=True) if isinstance(d_scale, torch.Tensor) else d_scale,
                                'u_scale': u_scale.to(device=original_out.device, non_blocking=True) if isinstance(u_scale, torch.Tensor) else u_scale,
                            }

                        cached, _ = self._lora_weight_cache.get_or_create(cache_key, create_int8_cache)
                        keys_to_release.add(cache_key)

                        chunked_int8_lora_forward(
                            x_2d, cached['d'], cached['u'],
                            cached['d_scale'], cached['u_scale'],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                    else:
                        curr_x = x_2d if x_2d.dtype == original_out.dtype else x_2d.to(dtype=original_out.dtype)

                        cache_key = (
                            "float",
                            _get_tensor_content_hash(d), _get_tensor_content_hash(u),
                            original_out.dtype, offset, size, original_out.shape[-1],
                        )

                        def create_float_cache():
                            return {
                                'd': d.to(device=original_out.device, dtype=original_out.dtype, non_blocking=True),
                                'u': u.to(device=original_out.device, dtype=original_out.dtype, non_blocking=True),
                            }

                        cached, _ = self._lora_weight_cache.get_or_create(cache_key, create_float_cache)
                        keys_to_release.add(cache_key)

                        chunked_lora_forward(
                            curr_x, cached['d'], cached['u'],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                
                except Exception as e:
                    print(f"Error computing LoRA patch: {e}")
                    continue

            if out is not None and any_patch_applied:
                if out is not original_out:
                    del original_out
                return out
            else:
                if out is not None and out is not original_out:
                    del out
                return original_out

        finally:
            for key in keys_to_release:
                try:
                    self._lora_weight_cache.release(key)
                except Exception:
                    pass


class WanLoRALoaderWithCLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
                "clip_type": (["STABLE_DIFFUSION", "STABLE_CASCADE", "SD3", "STABLE_AUDIO", "HUNYUAN_DIT", "FLUX", "MOCHI", "LTXV", "HUNYUAN_VIDEO", "PIXART", "COSMOS", "LUMINA2", "WAN", "HIDREAM", "CHROMA", "ACE", "OMNIGEN2", "QWEN_IMAGE", "HUNYUAN_IMAGE", "HUNYUAN_VIDEO_15", "OVIS", "KANDINSKY5", "KANDINSKY5_IMAGE", "NEWBIE", "FLUX2"], {"default": "WAN"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"

    def load_lora(self, model, lora_name, strength_model, strength_clip, clip_type, clip=None, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            free_mem, total_mem = torch.cuda.mem_get_info()
            if free_mem < total_mem * 0.1:
                print(f"Warning: Low GPU memory ({free_mem / 1e9:.2f}GB free). "
                      f"Consider reducing resolution or batch size.")

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model, clip)

        print(f"Loading LoRA: {lora_name}")
        offload_enabled = (offload_to_cpu == "enable")

        lora_state_dict = None
        lora_weights = None
        modules = None
        
        try:
            lora_state_dict = comfy.utils.load_torch_file(lora_path)
            lora_weights = parse_wan_lora(lora_state_dict, strength_model, debug=debug)
            
            if lora_state_dict is not None:
                del lora_state_dict
                lora_state_dict = None
            gc.collect()

            if lora_weights is None:
                print(f"Failed to parse LoRA: {lora_name}")
                return (model, clip)

            new_model = model.clone()
            
            with _object_patches_lock:
                if not hasattr(new_model, "object_patches"):
                    new_model.object_patches = {}
                else:
                    original_patches = new_model.object_patches
                    new_object_patches = {}
                    
                    for key, value in list(original_patches.items()):
                        if isinstance(value, LoRAWrapperModule):
                            value._lora_weight_cache.clear()
                            
                            try:
                                new_object_patches[key] = copy.deepcopy(value)
                            except Exception as e:
                                print(f"Warning: Rebuilding patch {key} due to deepcopy failure: {e}")
                                try:
                                    wrapped = value.wrapped_module
                                    patches = list(value.lora_patches) if value.lora_patches else []
                                    new_wrapper = LoRAWrapperModule(wrapped, patches)
                                    new_object_patches[key] = new_wrapper
                                except Exception as e2:
                                    print(f"Error: Could not rebuild patch {key}: {e2}")
                                    new_object_patches[key] = value
                        else:
                            try:
                                new_object_patches[key] = copy.deepcopy(value)
                            except Exception as e:
                                print(f"Warning: Failed to deepcopy patch {key}: {e}")
                                new_object_patches[key] = value
                    
                    new_model.object_patches = new_object_patches

            torch_model = new_model.model
            modules = dict(torch_model.named_modules())

            patched_count = 0
            patches_to_apply = []

            for key in lora_weights.weights:
                target_module = None
                target_key = None

                candidates = [key]
                
                if key.startswith("diffusion_model."):
                    candidates.append(key[len("diffusion_model."):])
                else:
                    candidates.append("diffusion_model." + key)

                var_candidates = []
                for c in candidates:
                    if ".self_attn." in c:
                        var_candidates.append(c.replace(".self_attn.", ".attn."))
                    elif ".attn." in c:
                        var_candidates.append(c.replace(".attn.", ".self_attn."))
                    if ".to_out.0" in c:
                        var_candidates.append(c.replace(".to_out.0", ".to_out"))
                    elif ".to_out" in c and ".to_out.0" not in c:
                        var_candidates.append(c.replace(".to_out", ".to_out.0"))

                candidates = candidates + var_candidates

                replacements = [
                    (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                    (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
                    (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv")
                ]

                final_pass_candidates = list(candidates)
                for c in candidates:
                    for old, new in replacements:
                        if old in c:
                            final_pass_candidates.append(c.replace(old, new))

                candidates = final_pass_candidates

                patch_offset = 0
                patch_size = 0

                for cand in candidates:
                    if cand in modules:
                        target_module = modules[cand]
                        target_key = cand

                        if ".qkv" in cand and isinstance(target_module, torch.nn.Linear):
                            total_out = target_module.weight.shape[0]
                            
                            if total_out % 3 != 0:
                                print(f"Warning: Cannot apply LoRA to QKV module {cand}: "
                                      f"output dimension {total_out} is not divisible by 3. "
                                      f"This LoRA patch will be skipped for key: {key}")
                                target_module = None
                                target_key = None
                                break
                            
                            head_dim = total_out // 3
                            
                            is_int8 = lora_weights.is_int8.get(key, False)
                            if is_int8:
                                down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                            else:
                                down, up, alpha = lora_weights.weights[key]
                            
                            lora_out_dim = up.shape[0] if up is not None and hasattr(up, 'shape') else 0
                            
                            if lora_out_dim == total_out:
                                patch_offset = 0
                                patch_size = 0
                                if debug:
                                    print(f"Full QKV LoRA detected for {key}: out_dim={lora_out_dim}")
                            elif lora_out_dim == head_dim:
                                component, is_fused = _identify_qkv_component(key)
                                
                                if component == 'q':
                                    patch_offset = 0
                                    patch_size = head_dim
                                elif component == 'k':
                                    patch_offset = head_dim
                                    patch_size = head_dim
                                elif component == 'v':
                                    patch_offset = head_dim * 2
                                    patch_size = head_dim
                                else:
                                    print(f"Warning: Could not determine Q/K/V type for partial LoRA key: {key}")
                                    target_module = None
                                    target_key = None
                            elif lora_out_dim > 0:
                                print(f"Warning: LoRA output dimension {lora_out_dim} doesn't match "
                                      f"expected full QKV ({total_out}) or single head ({head_dim}) for {key}")
                                target_module = None
                                target_key = None
                            break

                if target_module is not None and isinstance(target_module, torch.nn.Linear):
                    is_int8 = lora_weights.is_int8.get(key, False)
                    if is_int8:
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None

                    if hasattr(target_module, "weight"):
                        mod_out, mod_in = target_module.weight.shape

                        if patch_size > 0:
                            if patch_offset + patch_size > mod_out:
                                if debug:
                                    print(f"Warning: clamping patch for {key}")
                                patch_size = max(0, mod_out - patch_offset)

                        if mod_in != down.shape[1]:
                            if debug:
                                print(f"Skipping {key}: Input mismatch {mod_in} vs {down.shape[1]}")
                            continue

                    patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)
                    patches_to_apply.append((target_key, target_module, patch_tuple))

            wrappers_to_add = []
            for target_key, target_module, patch_tuple in patches_to_apply:
                current_patches = []
                if target_key in new_model.object_patches:
                    existing_obj = new_model.object_patches[target_key]
                    if hasattr(existing_obj, "lora_patches"):
                        current_patches = list(existing_obj.lora_patches)

                new_patch_list = current_patches + [patch_tuple]

                raw_module = target_module
                
                if isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module
                
                if target_key in new_model.object_patches:
                    existing = new_model.object_patches[target_key]
                    if isinstance(existing, LoRAWrapperModule):
                        raw_module = existing.wrapped_module
                    elif isinstance(existing, torch.nn.Linear):
                        raw_module = existing
                
                while isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module

                wrapper = LoRAWrapperModule(raw_module, new_patch_list)
                wrappers_to_add.append((target_key, wrapper))

            with _object_patches_lock:
                for target_key, wrapper in wrappers_to_add:
                    new_model.add_object_patch(target_key, wrapper)
                    patched_count += 1

            # CLIP passthrough - no LoRA applied to avoid OOM
            print(f"Applied {patched_count} LoRA patches (CLIP passthrough - no LoRA applied to CLIP).")
            return (new_model, clip)
            
        except Exception as e:
            print(f"Error loading LoRA {lora_name}: {e}")
            raise
            
        finally:
            if lora_state_dict is not None:
                del lora_state_dict
            if lora_weights is not None:
                del lora_weights
            if modules is not None:
                del modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
