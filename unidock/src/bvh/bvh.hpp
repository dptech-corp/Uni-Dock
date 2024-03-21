#include<vector>
#include<algorithm>
class Vector3 {
public:
    float x, y, z;

    Vector3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    Vector3 operator*(const Vector3& other) const {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }
};
// 射线类定义
class Ray {
public:
    Vector3 origin;    // 射线的原点
    Vector3 direction; // 射线的方向，应该是单位向量

    Ray(const Vector3& origin, const Vector3& direction) : origin(origin), direction(direction) {}
};

// 交点信息结构定义
struct Intersection {
    Vector3 point;   // 交点的位置
    Vector3 normal;  // 交点处的法向量
    float distance;  // 射线原点到交点的距离

    Intersection() : point(Vector3()), normal(Vector3()), distance(0) {}
};
class Geometry {
public:
    virtual ~Geometry() {}

    // 返回当前几何体的边界盒（AABB）
    virtual AABB boundingBox() const = 0;

    // 检测射线与当前几何体的相交，如果相交，返回true并更新交点信息
    virtual bool intersect(const Ray& ray, Intersection& intersection) const = 0;
};

class AABB {
public:
    Vector3 min; // 边界盒的最小坐标点
    Vector3 max; // 边界盒的最大坐标点
    Vector3 center; // 新增中心点属性
    AABB() : min(Vector3()), max(Vector3()) {}

    AABB(const Vector3& min, const Vector3& max) : min(min), max(max) {
        updateCenter(); // 初始化时更新中心点
    }
    void updateCenter() {
        center = (min + max) * 0.5f; // 计算并更新中心点
    }
    // 合并当前AABB和另一个AABB
    void merge(const AABB& other) {
        min.x = std::min(min.x, other.min.x);
        min.y = std::min(min.y, other.min.y);
        min.z = std::min(min.z, other.min.z);
        max.x = std::max(max.x, other.max.x);
        max.y = std::max(max.y, other.max.y);
        max.z = std::max(max.z, other.max.z);
        updateCenter(); // 合并后更新中心点
    }

    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
    const Vector3& dir = ray.direction;
    const Vector3& orig = ray.origin;

    Vector3 invD = Vector3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    Vector3 t0s = (min - orig) * invD;
    Vector3 t1s = (max - orig) * invD;

    Vector3 tMinVec = Vector3(std::min(t0s.x, t1s.x), std::min(t0s.y, t1s.y), std::min(t0s.z, t1s.z));
    Vector3 tMaxVec = Vector3(std::max(t0s.x, t1s.x), std::max(t0s.y, t1s.y), std::max(t0s.z, t1s.z));

    tMin = std::max({tMin, tMinVec.x, tMinVec.y, tMinVec.z});
    tMax = std::min({tMax, tMaxVec.x, tMaxVec.y, tMaxVec.z});

    if (tMax <= tMin) {
        return false;
    }

    return true;
}
};
// 简化的Sphere类定义
class Sphere : public Geometry {
public:
    Vector3 center;
    float radius;

    Sphere(const Vector3& c, float r) : center(c), radius(r) {}

    AABB boundingBox() const override {
        return AABB(center - Vector3(radius), center + Vector3(radius));
    }

    bool intersect(const Ray& ray, Intersection& intersection) const override {
        // 这里应该是球体与射线相交的实现，此处省略具体实现细节
        return false;
    }
};

struct BVHNode {
    AABB box; // 当前节点的边界盒
    BVHNode* left; // 左子节点
    BVHNode* right; // 右子节点
    std::vector<Geometry*> objects; // 仅叶子节点会持有几何体对象

    // 构造函数和析构函数
    BVHNode() : left(nullptr), right(nullptr) {}
    ~BVHNode() {
        delete left;
        delete right;
    }

    // 判断是否为叶子节点
    bool isLeaf() const { return !objects.empty(); }
};

#include <algorithm> // 用于std::sort

// 定义构建BVH的辅助函数
int partitionObjects(std::vector<Geometry*>& objects, int start, int end, const AABB& centroidBox) {
    // 示例：简单的中点分割方法
    float midpoint = 0.5f * (centroidBox.min.x + centroidBox.max.x);
    auto midIter = std::partition(objects.begin() + start, objects.begin() + end,
                                  [midpoint](const Geometry* obj) {
                                      return obj->boundingBox().center().x < midpoint;
                                  });
    int mid = std::distance(objects.begin(), midIter);
    return (mid == start || mid == end) ? (start + end) / 2 : mid; // 避免无限分割
}

// 实际的构建函数
BVHNode* buildBVH(std::vector<Geometry*>& objects, int start, int end) {
    if (start >= end) return nullptr; // 基本结束条件

    BVHNode* node = new BVHNode();

    // 计算当前对象集的边界盒和中心点盒
    AABB centroidBox;
    for (int i = start; i < end; ++i) {
        AABB objBox = objects[i]->boundingBox();
        node->box.merge(objBox);
        centroidBox.merge(objBox.center());
    }

    int objectCount = end - start;
    if (objectCount <= 2) { // 叶子节点条件
        node->objects.assign(objects.begin() + start, objects.begin() + end);
    } else {
        int mid = partitionObjects(objects, start, end, centroidBox);
        node->left = buildBVH(objects, start, mid);
        node->right = buildBVH(objects, mid, end);
    }

    return node;
}

