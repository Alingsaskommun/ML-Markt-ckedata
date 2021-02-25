import os
from os.path import join, basename

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)
from rastervision.pytorch_backend.examples.semantic_segmentation.utils import (
    example_multiband_transform, example_rgb_transform, imagenet_stats,
    Unnormalize)

# Nedan definieras sökväggar till in- och output av data
# Lägg till bilder, labels osv om ni vill ha fler/färre; 'x1','x2',...


def get_config(runner,
               multiband: bool = True,
               external_model: bool = False,
               augment: bool = False,
               nochip: bool = False,
               test: bool = False):
    root_uri = '/opt/data/output/'
    train_image_uris = ['/opt/data/data_input/images/1.tif','/opt/data/data_input/images/2.tif']
    train_label_uris = ['/opt/data/data_input/labels/1.geojson','/opt/data/data_input/labels/2.geojson']
    train_scene_ids = ['1','2']
    train_scene_list = list(zip(train_scene_ids, train_image_uris, train_label_uris))

    val_image_uri = '/opt/data/data_input/images/3.tif'
    val_label_uri = '/opt/data/data_input/labels/3.geojson'
    val_scene_id = '3'
  

    train_scenes_input = []
    
# channel_order: Första bandet är 0 (x-1). Om Dem är 4:e bandet är det nr 3. 
# Ska bara RGB analyseras blir channel_order = [0, 1, 2]

    if multiband:
        # Använd alla 4 channels
        channel_order = [0, 1, 2, 3]
        channel_display_groups = {'RGB': (0, 1, 2), 'elev': (3, )}
        aug_transform = example_multiband_transform
    else:
        # Använd endast red, green & elevation
        channel_order = [1, 0, 3]
        channel_display_groups = None
        aug_transform = example_rgb_transform

    if augment:
        mu, std = imagenet_stats['mean'], imagenet_stats['std']
        mu, std = mu[channel_order], std[channel_order]

        base_transform = A.Normalize(mean=mu.tolist(), std=std.tolist())
        plot_transform = Unnormalize(mean=mu, std=std)

        aug_transform = A.to_dict(aug_transform)
        base_transform = A.to_dict(base_transform)
        plot_transform = A.to_dict(plot_transform)
    else:
        aug_transform = None
        base_transform = None
        plot_transform = None

    chip_sz = 300
    img_sz = chip_sz
    if nochip:
        chip_options = SemanticSegmentationChipOptions()
    else:
        chip_options = SemanticSegmentationChipOptions(
            window_method=SemanticSegmentationWindowMethod.sliding,
            stride=chip_sz)

# Här definieras klasserna; byggnader och bakgrund i detta fallet.
    class_config = ClassConfig(
    names=['building', 'background'], colors=['red', 'black'])

    def make_scene(scene_id, image_uri, label_uri):
     
        raster_source = RasterioSourceConfig(
            uris=[image_uri],
            channel_order=channel_order,
            transformers=[StatsTransformerConfig()])
        vector_source = GeoJSONVectorSourceConfig(
            uri=label_uri, default_class_id=0, ignore_crs_field=True)
        label_source = SemanticSegmentationLabelSourceConfig(
            raster_source=RasterizedSourceConfig(
                vector_source=vector_source,
                rasterizer_config=RasterizerConfig(background_class_id=1)))
                
        # Vektor output skapas av byggnader class_id=0
        label_store = SemanticSegmentationLabelStoreConfig(
            rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])
        return SceneConfig(
            id=scene_id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)


    for scene in train_scene_list:
        train_scenes_input.append(make_scene(*scene))
        
    dataset = DatasetConfig(
    class_config=class_config,
    train_scenes=
        train_scenes_input
    ,
    validation_scenes=[
        make_scene(val_scene_id, val_image_uri, val_label_uri)
    ])
    
    
    # Chipsize, dvs storleken hur stora delar geotiffen ska delas upp i.  
    # Batch size kan vara så högt som möjligt utan att minnet går i taket. Öppna prestandafönstret under körningen och observera.
    # Antalet epoker är antalet gånger som inlärningsalgoritmen går genom hela träningsdatasetet.
    # För att använda en förtränad modell lägg till parametern 'init_weights' nedan; model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50, init_weights='/some/path/model.pth'),
    chip_sz = 300
    backend = PyTorchSemanticSegmentationConfig(
        data=SemanticSegmentationImageDataConfig(img_channels=len(channel_order), img_sz=img_sz),
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(lr=1e-4, num_epochs=2, batch_sz=4))
    chip_options = SemanticSegmentationChipOptions(window_method=SemanticSegmentationWindowMethod.sliding)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz)