#include "tnn/device/arm/acc/arm_add_layer_acc.h"
#include "tnn/interpreter/raw_buffer_mmap.h"

namespace TNN_NS {
class ArmGatherLayerAcc : public ArmLayerAcc {                                                            
public:                                                                                                            
    virtual ~ArmGatherLayerAcc(){};                                                                       
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);               

protected:
    RawBufferMMap rbm_data;
private:                                                                                                           
    template <typename T>                                                                                          
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                            
    DECLARE_ARM_FP16_LAYER_FUNC;                                                                                   
};
} //namespace TNN_NS