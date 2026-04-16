"""
阶梯电价配置模块
支持自定义阶梯电价设置
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


@dataclass
class ElectricityPricing:
    """
    阶梯电价配置类
    
    支持自定义时段和对应电价
    
    示例用法:
        # 默认电价
        pricing = ElectricityPricing()
        
        # 自定义电价
        pricing = ElectricityPricing(
            time_price_tiers=[
                (0, 20, 0.75),    # 0-20步: 谷电
                (20, 40, 1.15),   # 20-40步: 峰电
                (40, 50, 1.0),    # 40-50步: 平电
                (50, 70, 1.15),   # 50-70步: 峰电
            ]
        )
    """
    # 时段电价列表: [(起始时间, 结束时间, 电价), ...]
    # 默认电价（按分钟计）
    time_price_tiers: List[Tuple[int, int, float]] = field(default_factory=lambda: [
        (0, 20, 1.16),     # 平电
        (20, 140, 0.75),   # 谷电
        (140, 180, 0.46),  # 深谷
    ])
    
    # 电价类型名称
    tier_names: dict = field(default_factory=lambda: {
        0.46: "深谷",
        0.75: "谷电",
        1.16: "平电",
    })
    
    def get_price(self, time_step: int) -> float:
        """
        根据时间步获取当前电价
        
        Args:
            time_step: 当前时间步
            
        Returns:
            当前时段的电价
        """
        for start, end, price in self.time_price_tiers:
            if start <= time_step < end:
                return price
        # 默认返回最后一个时段的电价
        return self.time_price_tiers[-1][2] if self.time_price_tiers else 1.0
    
    def get_tier_name(self, price: float) -> str:
        """获取电价类型名称"""
        return self.tier_names.get(price, f"自定义({price})")
    
    def get_all_prices(self) -> List[float]:
        """获取所有不同的电价值"""
        return list(set(price for _, _, price in self.time_price_tiers))
    
    def get_max_price(self) -> float:
        """获取最高电价"""
        return max(price for _, _, price in self.time_price_tiers)
    
    def get_min_price(self) -> float:
        """获取最低电价"""
        return min(price for _, _, price in self.time_price_tiers)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'time_price_tiers': self.time_price_tiers,
            'tier_names': self.tier_names,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ElectricityPricing':
        """从字典创建实例"""
        return cls(
            time_price_tiers=[tuple(tier) for tier in data.get('time_price_tiers', [])],
            tier_names=data.get('tier_names', {}),
        )
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ElectricityPricing':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self):
        tiers_str = "\n".join([
            f"  [{start:2d}-{end:2d}): {price:.2f} ({self.get_tier_name(price)})"
            for start, end, price in self.time_price_tiers
        ])
        return f"ElectricityPricing:\n{tiers_str}"


# 预设电价方案
class PricingPresets:
    """预设电价方案"""
    
    @staticmethod
    def default() -> ElectricityPricing:
        """默认电价方案（按分钟计）"""
        return ElectricityPricing()
    
    @staticmethod
    def two_tier() -> ElectricityPricing:
        """两阶电价：峰谷电价"""
        return ElectricityPricing(
            time_price_tiers=[
                (0, 35, 0.6),     # 谷电
                (35, 70, 1.2),    # 峰电
            ],
            tier_names={0.6: "谷电", 1.2: "峰电"}
        )
    
    @staticmethod
    def three_tier() -> ElectricityPricing:
        """三阶电价：峰平谷"""
        return ElectricityPricing(
            time_price_tiers=[
                (0, 30, 1.04),     # 谷电
                (30, 90, 1.16),    # 平电
                (90, 180, 0.75),    # 峰电
            ],
            tier_names={1.04: "平电", 1.5: "峰电", 0.75: "谷电"}
        )
    
    @staticmethod
    def industrial() -> ElectricityPricing:
        """工业用电方案"""
        return ElectricityPricing(
            time_price_tiers=[
                (0, 15, 0.55),    # 深谷
                (15, 30, 0.85),   # 平段
                (30, 45, 1.25),   # 高峰
                (45, 55, 0.95),   # 平段
                (55, 70, 0.65),   # 谷段
            ],
            tier_names={0.55: "深谷", 0.65: "谷电", 0.85: "平段", 0.95: "平段", 1.25: "高峰"}
        )

    @staticmethod
    def daily_24h() -> ElectricityPricing:
        """24小时电价方案（按分钟计，区间边界对齐策略间隔）"""
        return ElectricityPricing(
            time_price_tiers=[
                (0, 420, 0.46),
                (420, 510, 0.75),
                (510, 630, 1.04),
                (630, 690, 1.16),
                (690, 960, 0.75),
                (960, 1140, 1.04),
                (1140, 1260, 1.16),
                (1260, 1380, 0.75),
                (1380, 1440, 0.46),
            ],
            tier_names={0.46: "深谷", 0.75: "谷电", 1.04: "平电", 1.16: "峰电"},
        )

    @staticmethod
    def constant() -> ElectricityPricing:
        """恒定电价方案"""
        return ElectricityPricing(
            time_price_tiers=[
                (0, 1440, 1.0),
            ],
            tier_names={1.0: "恒定"},
        )


if __name__ == "__main__":
    # 测试电价配置
    print("=== 默认电价方案 ===")
    default_pricing = PricingPresets.default()
    print(default_pricing)
    
    print("\n=== 测试获取电价 ===")
    for t in [0, 10, 25, 45, 60]:
        price = default_pricing.get_price(t)
        print(f"时间步 {t}: 电价 {price} ({default_pricing.get_tier_name(price)})")
    
    print("\n=== 自定义电价方案 ===")
    custom = ElectricityPricing(
        time_price_tiers=[
            (0, 30, 0.5),
            (30, 70, 1.5),
        ],
        tier_names={0.5: "低价时段", 1.5: "高价时段"}
    )
    print(custom)
